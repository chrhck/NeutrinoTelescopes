using NeutrinoTelescopes
using Flux
using CUDA
using Random
using StaticArrays
using BSON: @save, @load
using BSON
using CairoMakie
using Rotations
using LinearAlgebra
using DataFrames
using Zygote
using PoissonRandom
model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")

@load model_path model hparams opt tf_dict

Flux.testmode!(model)

pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0

begin
    pos = SA[0.0f0, 10.0f0, 20.0f0]
    dir_theta = deg2rad(50.0f0)
    dir_phi = deg2rad(50.0f0)
    dir = sph_to_cart(dir_theta, dir_phi)
    p = Particle(pos, dir, 0.0f0, Float32(1E5), PEMinus)

    target = MultiPMTDetector(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float32),
        UInt16(1)
    )


    wl_range = (300.0f0, 800.0f0)
    medium = make_cascadia_medium_properties(0.99f0)
    spectrum = CherenkovSpectrum(wl_range, 30, medium)

    source = ExtendedCherenkovEmitter(p, medium, wl_range)

    photon_setup = PhotonPropSetup([source], [target], medium, spectrum)
    photons = propagate_photons(photon_setup)

    calc_total_weight!(photons, photon_setup)
    calc_time_residual!(photons, photon_setup)

    rot = RotMatrix3(I)
    hits = make_hits_from_photons(photons, photon_setup, rot)
end

begin
    input = calc_flow_inputs([p], [target], tf_dict)
    times = -10:1:100
    log_pdf, log_expec = model(repeat(times, size(input, 2)), repeat(input, inner=(1, length(times))), true)
    log_pdf = reshape(log_pdf, length(times), size(input, 2),)
    log_expec = reshape(log_expec, length(times), size(input, 2),)
    fig = Figure(resolution=(1000, 700))
    ga = fig[1, 1] = GridLayout(4, 4)
    li = CartesianIndices((4, 4))
    for i in 1:16
        row, col = divrem(i - 1, 4)
        mask = hits[:, :pmt_id] .== i
        ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i")
        hist!(ax, hits[mask, :tres], bins=-10:5:100, weights=hits[mask, :total_weight], color=:orange, normalization=:density)
        lines!(ax, times, exp.(log_pdf[:, i] + log_expec[:, i]))

        @show sum(hits[mask, :total_weight]), exp.(log_expec[1, i])
    end


    fig
end

input = calc_flow_inputs([p], [target], tf_dict)
output = model.embedding(input)

flow_params = output[1:end-1, :]
log_expec = output[end, :]

expec = exp.(log_expec)
pois_expec = pois_rand.(expec)
mask = pois_expec .> 0

pois_expec[mask]
flow_params[:, mask]


function split_by(x, n)
    result = Vector{Vector{eltype(x)}}()
    start = firstindex(x)
    for l in n
        push!(result, x[start:start+l-1])
        start += l
    end

    return result
end

samples = sample_flow(flow_params[:, mask], model.range_min, model.range_max, pois_expec[mask])


size(iy)
hist(samples[11, :], bins=-10:5:100)



function likelihood(energy)
    p = Particle(pos, dir, 0.0f0, Float32(energy), PEMinus)
    input = calc_flow_inputs([p], [target], tf_dict)
    output = model.embedding(input)

    flow_params = output[1:end-1, :]
    log_expec = output[end, :]

    expec = exp.(log_expec)
    pois_expec = pois_rand.(expec)
    mask = pois_expec .> 0

    samples = sample_flow(flow_params[:, mask], model.range_min, model.range_max, pois_expec[mask])
    log_pdf, log_expec = model(repeat(times, size(input, 2)), repeat(input, inner=(1, length(times))), true)
end



Zygote.gradient()
