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
using StatsBase
using Base.Iterators
using Distributions
using Optim
using LogExpFunctions
using Base.Iterators

function poisson_logpmf(n, log_lambda)
    return n * log_lambda - exp(log_lambda) - loggamma(n + 1.0)
end


function sample_event(energy, dir_theta, dir_phi, position, model, tf_dict)
    
    dir = sph_to_cart(dir_theta, dir_phi)

    pmt_area = (75e-3 / 2)^2 * π
    target_radius = 0.21
    target = MultiPMTDetector(
        @SVector[0.0, 0.0, 0.0],
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float64),
        UInt16(1)
    )

    particle = Particle(position, dir, 0., energy, PEMinus)

    input = calc_flow_input(particle, target, tf_dict)
    
    output = model.embedding(input)

    flow_params = output[1:end-1, :]
    log_expec = output[end, :]

    expec = exp.(log_expec)

    n_hits = pois_rand.(expec)
    mask = n_hits .> 0

    non_zero_hits = n_hits[mask]
    return split_by(sample_flow(flow_params[:, mask], model.range_min, model.range_max, non_zero_hits), n_hits)
end


function likelihood(logenergy, dir_theta, dir_phi, position, samples, targets, model, tf_vec)
    
    n_pmt = get_pmt_count(eltype(targets))

    @assert length(targets)*n_pmt == length(samples)
    dir = sph_to_cart(dir_theta, dir_phi)

    energy = 10^logenergy
    particles = [ Particle(position, dir, 0., energy, PEMinus)]

    input = calc_flow_input(particles, targets, tf_vec)
    
    output::Matrix{eltype(input)} = model.embedding(input)

    flow_params = output[1:end-1, :]
    log_expec_per_source = output[end, :] # one per source and pmt

    log_expec = sum(reshape(log_expec_per_source, length(targets)*n_pmt, length(particles)), dims=2)[:, 1]

    rel_log_expec = log_expec_per_source .- log_expec

    hits_per = length.(samples)
    poiss = poisson_logpmf.(hits_per, log_expec)
    
    ix = LinearIndices((1:n_pmt*length(targets), eachindex(particles)))

    shape_llh = sum(
        LogExpFunctions.logsumexp(
            rel_log_expec[ix[i, j]] +
            sum(eval_transformed_normal_logpdf(
                samples[i],
                repeat(flow_params[:, ix[i, j]], 1, hits_per[i]),
                model.range_min,
                model.range_max))
            for j in eachindex(particles)
        )
        for i in 1:n_pmt*length(targets)
    )

    return sum(poiss) + shape_llh
end


function min_lh(samples, position, targets, model, tf_dict)

    function _func(x)
        logenergy, theta, phi = x
        return -likelihood(logenergy, theta, phi, position, samples, targets, model, tf_dict)
    end

    inner_optimizer = ConjugateGradient()
    lower = [2., 0, 0]
    upper = [5, π, 2*π]
    results  = optimize(_func, lower, upper, [3, 0.5, 0.5], Fminbox(inner_optimizer); autodiff=:forward)
    return results
end

function calc_resolution_maxlh(model, tf_dict, pos, n)
    
    min_vals =[]
    for _ in 1:n
        samples = sample_event(1E4, 0.1, 0.2, pos, model, tf_dict)
        res = min_lh(samples, pos,  [target], model,  tf_dict)    
        push!(min_vals, Optim.minimizer(res))
    end
    min_vals = reduce(hcat, min_vals)

    return min_vals
end


function convert(T, x::Particle)
    pos = T.(x.position)
    dir = T.(x.direction)
    energy = T(x.energy)
    time = T(x.time)

    return Particle(pos, dir, time, energy, x.type)
end

function convert(T, x::MultiPMTDetector)

    pos = T.(x.position)
    radius = T(x.radius)
    pmt_area = T(x.pmt_area)
    pmt_coordinates = T.(x.pmt_coordinates)

    return MultiPMTDetector(pos, radius, pmt_area, pmt_coordinates, x.module_id)
end




function mc_expectation(particles, targets)
    
    wl_range = (300.0f0, 800.0f0)
    medium = make_cascadia_medium_properties(0.99f0)
    spectrum = CherenkovSpectrum(wl_range, 30, medium)

    sources = [ExtendedCherenkovEmitter(convert(Float32, p), medium, wl_range) for p in particles]

    photon_setup = PhotonPropSetup(sources, convert.(Ref(Float32), targets), medium, spectrum, 1)
    photons = propagate_photons(photon_setup)

    calc_total_weight!(photons, photon_setup)
    calc_time_residual!(photons, photon_setup)

    rot = RotMatrix3(I)
    hits = make_hits_from_photons(photons, photon_setup, rot)
    return hits
end


function compare_mc_model(
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    models::Dict,
    hits)
    
    times = -10:1:100
    fig = Figure(resolution=(1000, 700))
    ga = fig[1, 1] = GridLayout(4, 4)

    for i in 1:16
        row, col = divrem(i - 1, 4)
        mask = hits[:, :pmt_id] .== i
        ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i")
        hist!(ax, hits[mask, :tres], bins=-10:5:100, weights=hits[mask, :total_weight], color=:orange, normalization=:density)
    end

    for (mname, model_path) in models
        @load model_path model hparams opt tf_vec
        input = calc_flow_input(particles, targets, tf_vec)
        log_pdf, log_expec = model(repeat(times, size(input, 2)), repeat(input, inner=(1, length(times))), true)
        log_pdf = reshape(log_pdf, length(times), size(input, 2),)
        log_expec = reshape(log_expec, length(times), size(input, 2),)
    
        for i in 1:16
            row, col = divrem(i - 1, 4)
            lines!(ga[col+1, row+1], times, exp.(log_pdf[:, i] + log_expec[:, i]), label=mname)
        end
    end

    fig
end

compare_mc_model(particles, targets, models) =  compare_mc_model(particles, targets, models, mc_expectation(particles, targets))


models = Dict(
    "1" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_1_FNL.bson"),
    "2" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_2_FNL.bson"),
    "3" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_3_FNL.bson"),
    "4" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_4_FNL.bson"),
    "5" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_5_FNL.bson")
)
    


pmt_area = (75e-3 / 2)^2 * π
target_radius = 0.21
target = MultiPMTDetector(
    @SVector[0.0, 0.0, 0.0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float64),
    UInt16(1)
)


begin
    pos = SA[-10., 10., 10.]
    dir_theta = 0.7
    dir_phi = 0.5
    dir = sph_to_cart(dir_theta, dir_phi)
    energy = 5e4
    particles = [ Particle(pos, dir, 0., energy, PEMinus)]

    hits = mc_expectation(particles, [target])
    compare_mc_model(particles, [target], models, hits)
end

50 / 1E6 * energy

begin
    pos = SA[-10., 10., 10.]
    dir_theta = 0.7
    dir_phi = 0.5
    dir = sph_to_cart(dir_theta, dir_phi)
    energy = 5e4

    delta = 50 / 1E6 * energy

    particles = [
         Particle(pos, dir, 0., energy, PEMinus),
         Particle(pos .+ dir .*delta, dir, delta*0.3, energy, PEMinus)
    ]

    hits = mc_expectation(particles, [target])
    compare_mc_model(particles, [target], models, hits)
end




model_res = Dict()
for (mname, model_path) in models
    @load model_path model hparams opt tf_dict
    Flux.testmode!(model)
    res = calc_resolution_maxlh(model, tf_dict, pos, 200)
    model_res[mname] = res
end


mean(model_res)

fig = Figure()
ax = Axis(fig[1, 1])

bins = 0:1:60

for (k, v) in model_res

    hist!(ax, rad2deg.(acos.(dot.(sph_to_cart.(v[2, :], v[3, :]), Ref(sph_to_cart(0.1, 0.2))))),
    label=k, bins=bins)

end

fig

@load models["4"] model hparams opt tf_dict
samples = sample_event(1E4, 0.1, 0.2, pos, model, tf_dict)

length(samples)
likelihood(4,  0.1, 0.2, pos, samples, [target], model, tf_dict)


log_energies = 3:0.05:5
lh_vals = [likelihood(e,  0.1, 0.2, pos, samples, [target], model, tf_dict) for e in log_energies]
CairoMakie.scatter(log_energies, lh_vals)

zeniths = 0:0.01:0.5
lh_vals = [likelihood(4,  z, 0.2, pos, samples, [target], model, tf_dict) for z in zeniths]
CairoMakie.scatter(zeniths, lh_vals)




@code_warntype calc_resolution_maxlh(model, tf_dict, pos, 200)


pos = SA[10., 30., 10.]
samples = sample_event(1E4, 0.1, 0.2, pos, tf_dict)

log_energies = 3:0.05:5
lh_vals = [likelihood(e,  0.1, 0.2, pos, samples, [target], tf_dict) for e in log_energies]
scatter(log_energies, lh_vals, axis=(limits=(3, 5, -10000, -500), ))

zeniths = 0:0.01:0.5
lh_vals = [likelihood(4,  z, 0.2, pos, samples, [target], tf_dict) for z in zeniths]
scatter(zeniths, lh_vals)

Zygote.gradient( x -> likelihood(x[1], x[2], x[3], pos, samples, [target], tf_dict), [1E4, 0.1, 0.2])



hist(rad2deg.(acos.(dot.(sph_to_cart.(min_vals[2, :], min_vals[3, :]), Ref(sph_to_cart(0.1, 0.2))))))







function calc_fisher(logenergy, dir_theta, dir_phi, n, model; use_grad=false)
    pmt_area = (75e-3 / 2)^2 * π
    target_radius = 0.21
    target = MultiPMTDetector(
        @SVector[0.0, 0.0, 0.0],
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float64),
        UInt16(1)
    )
   
    matrices = []
    for _ in 1:n

        pos_theta = acos(rand(Uniform(-1, 1)))
        pos_phi = rand(Uniform(0, 2*pi))
        r = sqrt(rand(Uniform(5^2, 50^2)))
        pos = r .* sph_to_cart(pos_theta, pos_phi)

        for __ in 1:20
            samples = sample_event(10^logenergy, dir_theta, dir_phi, pos, model, tf_dict)
            if use_grad
                logl_grad = collect(Zygote.gradient(
                    (logenergy, dir_theta, dir_phi) -> likelihood(logenergy, dir_theta, dir_phi, pos, samples, [target], model, tf_dict),
                    logenergy, dir_theta, dir_phi))

                push!(matrices, logl_grad .* logl_grad')
            else
                logl_hessian =  Zygote.hessian(
                    x -> likelihood(x[1], x[2], x[3], pos, samples, [target], model, tf_dict),
                    [logenergy, dir_theta, dir_phi])
                push!(matrices, .-logl_hessian)
            end
        end
    end

    return mean(matrices)
end





logenergies = 2:0.5:5
sds= [calc_fisher(e, 0.1, 0.2, 50, model, use_grad=true) for e in logenergies]
cov = inv.(sds)

sampled_sds = []
for c in cov

    cov_za = c[2:3, 2:3]
    dist = MvNormal([0.1, 0.2], 0.5 * (cov_za + cov_za'))
    rdirs = rand(dist, 10000)

    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(sph_to_cart(0.1, 0.2)))))
    push!(sampled_sds, std(dangles))
end



CairoMakie.scatter(logenergies, Vector{Float64}(sampled_sds))





zazres =  (reduce(hcat, [sqrt.(v) for v in diag.(inv.(sds))])[2:3, :])

zazi_dist = MvNormal()







rad2deg.(acos.(dot.(sph_to_cart.(v[2, :], v[3, :]), Ref(sph_to_cart(0.1, 0.2))))


CairoMakie.scatter(logenergies, reduce(hcat, [sqrt.(v) for v in diag.(inv.(sds))])[1, :], axis=(yscale=log10, ))

CairoMakie.scatter(logenergies, rad2deg.(reduce(hcat, [sqrt.(v) for v in diag.(inv.(sds))])[2, :]))


logl_grad = Zygote.gradient(
    energy -> likelihood(energy, samples, [target], tf_dict),
    1E4)


a = Enzyme.autodiff(Enzyme.Reverse, test, Active, Active(1E4), samples, [target], tf_dict)
#@show a

pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0


pos = SA[0.0f0, 10.0f0, 30.0f0]
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
   

input = calc_flow_inputs([p], [target], tf_dict)
output = model.embedding(input)

flow_params = output[1:end-1, :]
log_expec = output[end, :]

expec = exp.(log_expec)
pois_expec = pois_rand.(expec)
mask = pois_expec .> 0
