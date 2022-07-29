using Revise
using BenchmarkTools

using Plots
using StatsPlots
using Parquet
using StaticArrays
using Unitful
using LinearAlgebra
using Distributions
using Base.Iterators
using Random
using StatsFuns
using LogExpFunctions
using DataFrames
using CUDA
using Flux
using BSON: @save, @load
using BSON
using Zygote


using NeutrinoTelescopes
using .Spectral
using .Medium
using .LightYield
using .Emission
using .Detection
using NeutrinoTelescopes.Modelling


const PROJECT_ROOT = pkgdir(NeutrinoTelescopes)

rng = MersenneTwister(31338)
params = Dict(
    :width => 1024,
    :learning_rate => 0.0009,
    :batch_size => 1024,
    :data_file => joinpath(PROJECT_ROOT, "assets/photon_fits.parquet"),
    :dropout_rate => 0.5,
    :rng => rng,
    :epochs => 400
)



model = BSON.load(joinpath(PROJECT_ROOT, "assets/photon_model.bson"), @__MODULE__)[:model] |> gpu

typeof(model)

train_data, test_data, trafos = get_data(Modelling.Hyperparams(; params...))

output_trafos = [
    trafos[(:fit_alpha, :log_fit_alpha)],
    trafos[(:fit_theta, :log_fit_theta)],
    trafos[(:det_fraction, :log_det_fraction_scaled)]
]




function extended_llh_per_module(
    x::AbstractVector{T},
    poisson::Sampleable,
    shape::Sampleable) where {T<:Real}
    n_obs = size(x, 1)

    pllh = loglikelihood(poisson, n_obs)
    sllh = loglikelihood.(shape, x)

    return pllh + sum(sllh)

end

function extended_llh_per_module(
    x::AbstractVector{U},
    targets::AbstractVector{V},
    particle::LightYield.Particle,
    medium::Medium.MediumProperties,
    model::Flux.Chain) where {T<:Real,V<:Detection.PhotonTarget,U<:AbstractVector{T}}


    predictions, _ = evaluate_model(targets, particle, medium, model)
    shape = shape_mixture_per_module(predictions)
    poisson = poisson_dist_per_module(predictions, sources)


    extended_llh_per_module.(x, poisson, shape)

end


function sample_event(
    poissons::AbstractVector{U},
    shapes::AbstractVector{V},
    sources::AbstractVector{W}) where {U<:Sampleable,V<:Sampleable,W<:CherenkovSegment}

    if size(poissons) != size(shapes) != size(sources)
        error("Vectors have to be of same size")
    end


    event = Vector{Vector{Float64}}(undef, size(poissons))
    for i in 1:size(poissons, 1)
        n_ph = rand(poissons[i])
        event[i] = rand(shapes[i], n_ph) .= sources[i].time
    end

    event
end



zenith_angle = 20
azimuth_angle = 100

positions = Detection.make_detector_cube(5, 5, 10, 50.0, 100.0)
targets = make_targets(positions)
particle = LightYield.Particle(
    @SVector[0.0, 0.0, 20.0],
    sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
    100.0,
    1E5,
    LightYield.EMinus
)

medium64 = make_cascadia_medium_properties(Float64)

poissons = poisson_dist_per_module(targets, particle, medium64, 0.5, model)
shapes = shape_mixture_per_module(targets, particle, medium64, 0.5, model)


event = sample_event(poissons, shapes)

struct LossFunction{T}
    x::T
    y::T
    z::T
    theta::T
    phi::T
    log_energy::T
end


function eval_loggamma_mix(logweights, αs, θs, x)
    gamma_evals = gammalogpdf.(αs, θs, x)
    LogExpFunctions.logsumexp(logweights .+ gamma_evals)
end



function (l::LossFunction{T})(
    target::Detection.PhotonTarget{T},
    times::Vector{T},
    int_grid::AbstractVector{T},
    medium::MediumProperties,
    trafos::AbstractVector{Symbol},
    model::Flux.Chain) where {T<:Real}

    targets = [target]
    nph = [length(times)]
    event = [times]

    n_targets = 1
    n_sources = length(int_grid) - 1

    position = SA[l.x, l.y, l.z]
    direction = sph_to_cart(l.theta, l.phi)
    time = 0.0

    particle = LightYield.Particle(
        position,
        direction,
        time,
        exp10(l.log_energy),
        LightYield.EMinus
    )

    source_out = Vector{LightYield.CherenkovSegment{T}}(undef, n_sources)

    source_out_buf = Zygote.Buffer(source_out)
    particle_to_elongated_lightsource!(
        particle,
        int_grid,
        medium,
        (300.0, 800.0),
        source_out_buf)
    source_out::Vector{LightYield.CherenkovSegment{T}} = copy(source_out_buf)


    inputs = Matrix{Float32}(undef, (2, n_targets * n_sources))
    inp_buf = Zygote.Buffer(inputs)

    @inbounds for (i, (source, target)) in enumerate(product(source_out, targets))
        res::Tuple{Float32,Float32} = source_to_input(source, target)
        inp_buf[1, i] = res[1]
        inp_buf[2, i] = res[2]
    end

    inputs_gpu::CuMatrix{Float32} = gpu(copy(inp_buf))
    predictions::CuMatrix{Float32} = model(inputs_gpu)

    αs = cpu(reshape(exp.(predictions[1, :]), (n_sources, n_targets)))
    θs = cpu(reshape(exp.(predictions[2, :]), (n_sources, n_targets)))
    det_fractions = reshape(exp10.(.-(predictions[3, :] .* 10 .+ 3)), (n_sources, n_targets))

    #= poissons = poisson_dist_per_module(predictions_rshp, source_out)
    shapes = shape_mixture_per_module(predictions_rshp)


    shape_lh = [nph[i] > 0 ? loglikelihood(shapes[i], event[i]) : 0.0 for i in 1:n_targets]
    sum(shape_lh .+ loglikelihood.(poissons, nph)) =#


    log_weighting::Matrix{Float32} = cpu(log.(det_fractions ./ sum(det_fractions, dims=1))) # n_sources * n_targetrs

    src_ph = gpu([src.photons for src in source_out])
    nph_expec = sum(det_fractions .* src_ph, dims=1)

    nph = gpu(nph)
    lh_sum::Float32 = cpu(sum(poislogpdf.(nph_expec, nph)))

    @inbounds for j in 1:n_targets
        if nph[j] > 0
            #log_weighting = log.(det_fractions[:, j] ./ sum(det_fractions[:, j]))

            lh_sum += sum(eval_loggamma_mix.((log_weighting,), (αs[:, j],), (θs[:, j],), event[j]))
        end

        #nph_expec = [sum([det_fractions[i, j] * source_out[i].photons for i in 1:n_sources]) for j in 1:n_targets]
        #lh_sum += sum(poislogpdf.(nph_expec, nph))
    end
    cpu(lh_sum)

end


function eval_fisher(particle, targets, event, precision, medium, output_trafos, model)

    len_range = (0.0, 20.0)

    int_grid = range(len_range[1], len_range[2], step=precision)
    n_steps = size(int_grid, 1)

    ptheta = acos(particle.direction[3])
    pphi = acos(particle.direction[1] / sin(ptheta))
    lfunc = LossFunction(
        particle.position.x,
        particle.position.y,
        particle.position.z,
        ptheta,
        pphi,
        log10(particle.energy)
    )

    #[lfunc(target, even, int_grid, medium, output_trafos, model) for (target, even) in zip(targets, event)]
    gradients = [Zygote.gradient(m -> m(target, times, int_grid, medium, output_trafos, model), lfunc)[1] for (target, times) in zip(targets, event)]

    gradients_df = DataFrame(gradients)
    grad_vec = reshape(sum.(eachcol(gradients_df)), (1, 6))
    fisher_info = grad_vec .* grad_vec'

end


eval_fisher(particle, targets, event, 0.5, medium64, output_trafos, model)

