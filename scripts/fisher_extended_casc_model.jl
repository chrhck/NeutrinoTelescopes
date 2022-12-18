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
using SpecialFunctions
using Enzyme
model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")

@load model_path model hparams opt tf_dict

Flux.testmode!(model)





function poisson_logpmf(n, log_lambda)
    return n * log_lambda - exp(log_lambda) - loggamma(n + 1.0)
end

function sum_llh_per_pmt!(log_pdf, log_expec, n_hits_per_pmt, out)
    nz_ix = firstindex(log_pdf)
    for i in eachindex(n_hits_per_pmt)
        nhits = n_hits_per_pmt[i]
        out[i] = poisson_logpmf(nhits, log_expec[i])

        if nhits > 0
            @views shape_lh = sum(log_pdf[nz_ix:nz_ix+nhits-1])
            out[i] += shape_lh
            nz_ix += nhits
        end
    end
end

function sum_llh_per_pmt(log_pdf, log_expec, n_hits_per_pmt)
    out = similar(log_expec)
    sum_llh_per_pmt!(log_pdf, log_expec, n_hits_per_pmt, out)
    return out
end

function sum_llh_per_pmt_sum(log_pdf, log_expec, n_hits_per_pmt)
    out = zero(eltype(log_pdf))
    nz_ix = firstindex(log_pdf)
    for i in eachindex(n_hits_per_pmt)
        nhits = n_hits_per_pmt[i]
        out += poisson_logpmf(nhits, log_expec[i])

        if nhits > 0
            @views shape_lh = sum(log_pdf[nz_ix:nz_ix+nhits-1])
            out += shape_lh
            nz_ix += nhits
        end
    end
    return out
end


function sample_event(energy)
    pos = SA[0.0f0, 10.0f0, 20.0f0]
    dir_theta = deg2rad(50.0f0)
    dir_phi = deg2rad(50.0f0)
    dir = sph_to_cart(dir_theta, dir_phi)

    pmt_area = Float32((75e-3 / 2)^2 * π)
    target_radius = 0.21f0
    target = MultiPMTDetector(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float32),
        UInt16(1)
    )

    p = Particle(pos, dir, 0.0f0, Float32(energy), PEMinus)
    input = calc_flow_inputs([p], [target], tf_dict)
    output = model.embedding(input)

    flow_params = output[1:end-1, :]
    log_expec = output[end, :]

    expec = exp.(log_expec)
    
    n_hits = pois_rand.(expec)
    mask = n_hits .> 0

    non_zero_hits = n_hits[mask]
    return split_by(sample_flow(flow_params[:, mask], model.range_min, model.range_max, non_zero_hits), n_hits)
end




function likelihood(energy, samples, targets, tf_dict)
    pos = SA[0.0f0, 10.0f0, 20.0f0]
    dir_theta = deg2rad(50.0f0)
    dir_phi = deg2rad(50.0f0)
    dir = sph_to_cart(dir_theta, dir_phi)
  

    p = Particle(pos, dir, 0.0f0, Float32(energy), PEMinus)
    @code_warntype( calc_flow_inputs([p], targets, tf_dict))
    input = calc_flow_inputs([p], targets, tf_dict)
    #=
    output = model.embedding(input)

    flow_params = output[1:end-1, :]
    log_expec = output[end, :]

    samp_cat = reduce(vcat, samples)
    hits_per = length.(samples)
    fp_rep = repeat_for(flow_params, hits_per)
    poiss = poisson_logpmf.(hits_per, log_expec)
    shape_per = sum.(split_by(
        eval_transformed_normal_logpdf(samp_cat, fp_rep, model.range_min, model.range_max),
        hits_per))

    
    lh = sum(poiss .+ shape_per)
       #=
    lh = zero(energy)
    for (samp, fp, lep) in zip(samples, eachcol(flow_params), log_expec)
        shape_lh = 0
     
        fp = reshape(fp, length(fp), 1)
        for s in samp
            shape_lh += eval_transformed_normal_logpdf([s], fp, model.range_min, model.range_max)[1]
        end
        

        shape_lh = sum(eval_transformed_normal_logpdf(samp, fp_rep, model.range_min, model.range_max))

        poiss = poisson_logpmf(length(samp), lep)

        lh += poiss + sum(shape_lh)
    end
    =#
    =#

    return input[1]


    return lh
end

pmt_area = (75e-3 / 2)^2 * π
target_radius = 0.21
target = MultiPMTDetector(
    @SVector[0.0, 0.0, 0.0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float64),
    UInt16(1)
)


samples = sample_event(1E4)

likelihood(1E4, samples, [target], tf_dict)


#a = Enzyme.autodiff(Enzyme.Reverse, likelihood, Active, Active(1E4), samples, [target], tf_dict)
#@show a

pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0


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

    photon_setup = PhotonPropSetup([source], [target], medium, spectrum, 1)
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
