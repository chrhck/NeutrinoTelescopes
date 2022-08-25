module Modelling

using Sobol
using Distributions
using ProgressLogging
using Parquet
using DataFrames
using Flux
using CUDA
using Base.Iterators: partition
using Flux: params as fparams
using Flux.Data: DataLoader
using Flux.Losses: mse
using Flux: @epochs
using Random
using LinearAlgebra
using Base.Iterators
using Zygote
using ParameterSchedulers
using ParameterSchedulers: AbstractSchedule
using Unitful
using PhysicalConstants.CODATA2018
using StaticArrays
using Base.Iterators
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_increment, set_step!, set_step_increment!

using ...Types
using ...Utils
using ..Detection
using ..Medium
using ..PhotonPropagationCuda
using ..LightYield
using ..Spectral


export get_dir_reweight, fit_photon_dist, make_photon_fits
export Hyperparams, get_data
export splitdf, read_from_parquet
export loss_all, train_mlp
export source_to_input
export apply_transformation, reverse_transformation, transform_model_output!
export poisson_dist_per_module, shape_mixture_per_module, evaluate_model, sample_event
export NoSchedulePars, SinDecaySchedulePars, LRScheduleParams

global c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)


function get_dir_reweight(em_dir::SVector{3, T}, shower_axis::SVector{3, U}, ref_ix::T) where {T<:Real, U<:Real}
    # Assume that source-target direction is e_z    
    rot_ph_dir = rot_to_ez_fast(shower_axis, em_dir)

    ph_cos_theta = rot_ph_dir[3]
    norm = cherenkov_ang_dist_int(ref_ix) .* 2
    
    cherenkov_ang_dist(ph_cos_theta, ref_ix) / norm
end


function fit_photon_dist(obs_photon_df, n_ph_gen)
   
    ph_abs_weight = obs_photon_df[:, :abs_weight]
    ph_tres = obs_photon_df[:, :tres]

    pmt_acc_weight = p_one_pmt_acc.(obs_photon_df[:, :wavelength])
    total_weight = ph_abs_weight .* pmt_acc_weight

    mask = ph_tres .>= 0

    try
        dfit = fit_mle(Gamma, ph_tres[mask], total_weight[mask])
        dfit.α, dfit.θ, sum(total_weight) / n_ph_gen
    catch e
        #=nanmask = isnan.(total_weight[mask])
        @show total_weight[mask][nanmask]
        @show ph_abs_weight[mask][nanmask]
        @show pmt_acc_weight[mask][nanmask]

        tot_mask = mask .&& nanmask

        ixs = collect(1:nrow(obs_photon_df))
        @show obs_photon_df[ixs[tot_mask], :]

        nanmask = isnan.(ph_tres[mask])
        @show ph_tres[mask][nanmask]

        @show suffstats(Gamma, ph_tres[mask], total_weight[mask])
        @show ph_tres[mask][1:10]
        @show total_weight[mask][1:10]
        =#
        return (NaN, NaN, NaN)
    end
    
    
    

end

"""
    make_photon_fits(n_photons_per_dist::Int64, n_distances::Integer, n_angles::Integer)

Convenience function for propagating photons and fitting the arrival time distributions.
"""
function make_photon_fits(n_photons_per_dist::Int64, max_nph_det::Int64, n_distances::Integer, n_angles::Integer, max_dist::Float32=300.0f0)

    s = SobolSeq([0.0], [pi])
    medium = make_cascadia_medium_properties(Float32)

    s2 = SobolSeq([0.0f0], [Float32(log10(max_dist))])
    distances = 10 .^ reduce(hcat, next!(s2) for i in 1:n_distances)

    results_fit = DataFrame(
        fit_alpha=Float64[],
        fit_theta=Float64[],
        det_fraction=Float64[],
        obs_angle=Float64[],
        distance=Float64[]
        )

    

    @progress name = "Propagating photons" for dist in distances
        obs_angles = reduce(hcat, next!(s) for i in 1:n_angles)
        for obs_angle in obs_angles

            direction = sph_to_cart(Float32(obs_angle), 0f0)


            nph_this = n_photons_per_dist
            prop_res = nothing
            nph_sim = nothing
            
            while true
                source = PointlikeCherenkovEmitter(SA[0f0, 0f0, 0f0], direction, 0f0, nph_this, CherenkovSpectrum((300f0, 800f0), 40, medium))
                prop_res, nph_sim = propagate_source(source, dist, medium)

                if nrow(prop_res) > 10
                    break
                end
                nph_this *= 10

                if nph_this > 1E12
                    @warn "No photons detected after propagating 1E12, skipping" dist obs_angle
                    break
                end
            end

            if nrow(prop_res) <= 10
                continue
            end

            # if we have more detected photons than we want, discard und upweight the rest
            if nrow(prop_res) > max_nph_det
                upweight = nrow(prop_res) / max_nph_det
                prop_res = prop_res[1:max_nph_det, :]
                prop_res[:, :abs_weight] .*= upweight
            end

            fit_result = fit_photon_dist(prop_res, nph_sim)
    
            if fit_result[1] != NaN
                push!(results_fit, (fit_alpha=fit_result[1], fit_theta=fit_result[2], det_fraction=fit_result[3], obs_angle=obs_angle, distance=dist))
            end
        end
    end
    results_fit
end


abstract type LRScheduleParams end

Base.@kwdef mutable struct NoSchedulePars <: LRScheduleParams
    learning_rate::Float64
end

Base.@kwdef mutable struct SinDecaySchedulePars <: LRScheduleParams
    lr_max::Float64
    lr_min::Float64
    lr_period::Int64
end


struct NoSchedule{T<:Number} <: AbstractSchedule{false}
    λ::T
end
(schedule::NoSchedule)(t) = schedule.λ 


make_scheduler(pars::NoSchedulePars) = NoSchedule(pars.learning_rate)
make_scheduler(pars::SinDecaySchedulePars) = SinDecay2(λ0=pars.lr_min, λ1=pars.lr_max, period=pars.lr_period)


Base.@kwdef mutable struct Hyperparams
    data_file::String
    batch_size::Int64
    lr_schedule_pars::LRScheduleParams 
    epochs::Int64
    width::Int64
    dropout_rate::Float64
    seed::Int64
    tblogger = true
    savepath = "runs/"
end

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

apply_transformation(x::Real, ::Val{:neg_log}) = -log10(x)
reverse_transformation(x::Real, ::Val{:neg_log}) = exp10(-x)

apply_transformation(x::Real, ::Val{:cos}) = cos(x)
reverse_transformation(x::Real, ::Val{:cos}) = acos(x)

apply_transformation(x::Real, ::Val{:neg_log_scale}) = (-log10(x) - 3) / 10
reverse_transformation(x::Real, ::Val{:neg_log_scale}) = exp10(-(x * 10 + 3))

apply_transformation(x::Real, ::Val{:log}) = log10(x)
reverse_transformation(x::Real, ::Val{:log}) = exp10(x)


apply_transformation(x::U, t::Val) where {T<:Real,U<:AbstractVector{T}} = apply_transformation.(x, Ref(t))
reverse_transformation(x::U, t::Val) where {T<:Real,U<:AbstractVector{T}} = reverse_transformation.(x, Ref(t))


function read_from_parquet(filename, trafos)
    results_df = DataFrame(read_parquet(filename))

    results_df[!, :] = convert.(Float32, results_df[!, :])

    transform!(results_df, [in => (x -> apply_transformation(x, Val(value))) => out for ((in, out), value) in trafos]...)

    #=
    results_df[!, :log_det_fraction] = -log10.(results_df[!, :det_fraction])
    results_df[!, :log_det_fraction_scaled] = ((-log10.(results_df[!, :det_fraction])) .- 3) ./ 10

    results_df[!, :log_distance] = log10.(results_df[!, :distance])
    results_df[!, :cos_obs_angle] = cos.(results_df[!, :obs_angle])
    results_df[!, :fit_alpha_scaled] = results_df[!, :fit_alpha] ./ 100

    =#
    feature_names = [:log_distance, :cos_obs_angle]
    target_names = [:log_fit_alpha, :log_fit_theta, :neg_log_det_fraction_scaled]

    df_train, df_test = splitdf(results_df, 0.8)

    features_train = Matrix{Float32}(df_train[:, feature_names])'
    targets_train = Matrix{Float32}(df_train[:, target_names])'
    features_test = Matrix{Float32}(df_test[:, feature_names])'
    targets_test = Matrix{Float32}(df_test[:, target_names])'

    return (features_train, targets_train, features_test, targets_test)
end

function get_data(args::Hyperparams)

    trafos = Dict(
        (:det_fraction, :neg_log_det_fraction) => :neg_log,
        (:det_fraction, :neg_log_det_fraction_scaled) => :neg_log_scale,
        (:obs_angle, :cos_obs_angle) => :cos,
        (:fit_alpha, :log_fit_alpha) => :log,
        (:fit_theta, :log_fit_theta) => :log,
        (:distance, :log_distance) => :log
    )

    features_train, targets_train, features_test, targets_test = read_from_parquet(args.data_file, trafos)

    rng = MersenneTwister(args.seed)
    loader_train = DataLoader((features_train, targets_train), batchsize=args.batch_size, shuffle=true, rng=rng)
    loader_test = DataLoader((features_test, targets_test), batchsize=args.batch_size)

    loader_train, loader_test, trafos
end

function loss_all(dataloader, model)
    l = 0.0f0
    for (x, y) in dataloader
        l += mse(model(x), y)
    end
    l / length(dataloader)
end


function train_mlp(; kws...)
    ## Initialize hyperparameter arguments
    args = Hyperparams(; kws...)

    ## Load processed data
    train_data, test_data, trafos = get_data(args)
    train_data = gpu.(train_data)
    test_data = gpu.(test_data)


    model = Chain(
        Dense(2, args.width, relu, init=Flux.glorot_uniform),
        Dense(args.width, args.width, relu, init=Flux.glorot_uniform),
        Dropout(args.dropout_rate),
        Dense(args.width, args.width, relu, init=Flux.glorot_uniform),
        Dropout(args.dropout_rate),
        Dense(args.width, args.width, relu, init=Flux.glorot_uniform),
        Dropout(args.dropout_rate),
        Dense(args.width, 3))


    model = gpu(model)
    loss(x, y) = mse(model(x), y)
    optimiser = ADAM()
    schedule = make_scheduler(args.lr_schedule_pars) 


    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_increment)
        set_step_increment!(tblogger, 0) ## 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    function report(epoch)
        train_loss = loss_all(train_data, model)
        test_loss = loss_all(test_data, model)
        #println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss = train_loss
                @info "test" loss = test_loss

            end
        end
    end

    ps = Flux.params(model)

    @info "Starting training."
    Flux.trainmode!(model)

    for (eta, epoch) in zip(schedule, 1:args.epochs)
        optimiser.eta = eta

        for (x, y) in train_data

            gs = Flux.gradient(ps) do
                loss(x, y)
            end

            Flux.Optimise.update!(optimiser, ps, gs)
        end
        report(epoch)
    end

    #@epochs args.epochs Flux.train!(loss, fparams(model), train_data, optimiser, cb=evalcb)
    Flux.testmode!(model)
    return model, train_data, test_data, trafos
end

function source_to_input(source::PointlikeCherenkovEmitter, target::PhotonTarget)
    em_rec_vec = target.position .- source.position
    distance = norm(em_rec_vec)
    em_rec_vec = em_rec_vec ./ distance
    cos_obs_angle = dot(em_rec_vec, source.direction)

    return Float32(log10(distance)), Float32(cos_obs_angle)

end

function source_to_input(sources::AbstractVector{<:PointlikeCherenkovEmitter}, targets::AbstractVector{<:PhotonTarget})

    total_size = size(sources, 1) * size(targets, 1)

    inputs = Matrix{Float32}(undef, (2, total_size))

    for (i, (src, tgt)) in enumerate(product(sources, targets))
        inputs[:, i] .= source_to_input(src, tgt)
    end
    inputs
end

function transform_model_output!(output::Union{Zygote.Buffer,AbstractMatrix{T}}, trafos::AbstractVector{Symbol}) where {T<:Real}

    if size(output, 1) != size(trafos, 1)
        error("Feature dimension size must equal number of transformations")
    end

    output .= reverse_transformation.(output, Val.(trafos))

end

function evaluate_model(
    targets::AbstractVector{U},
    particle::Particle,
    medium::MediumProperties,
    precision::T,
    model::Flux.Chain,
    trafos::AbstractVector{Symbol},
    max_dist::Number = 300
) where {U<:Detection.PhotonTarget,T<:Real}

    sources = particle_to_elongated_lightsource(
        particle,
        (T(0.0), T(20.0)),
        precision,
        medium,
        (T(300.0), T(800.0)),
    )

    inputs = source_to_input(sources, targets)

    log10_max_dist = log10(max_dist)

    mask = (inputs[1, :] .<= log10_max_dist) .&& (inputs[1, :] .>= 0)
    
    predictions::Matrix{Float32} = cpu(model(gpu(inputs)))
    Modelling.transform_model_output!(predictions, trafos)

    predictions_rshp = reshape(predictions, (3, size(sources, 1), size(targets, 1)))
    
    predictions_rshp[3, :, :] = mapslices(slice -> slice .* area_acceptance.(targets), predictions_rshp[3, :, :], dims=2) 
    
    mask_rshp = reshape(mask, (length(sources), length(targets)))

    #=
    distances = empty((length(sources), length(targets)))

    for (i, j) in zip(eachindex(sources), eachindex(targets))
        distances[i, j] = norm(sources[i].position - targets[j].position)
    end
    =#

    distances = reshape(exp10.(inputs[1, :]),  (length(sources), length(targets)))

    return predictions_rshp, sources, mask_rshp, distances
end

function shape_mixture_per_module(
    params::AbstractArray{U,3},
    sources::AbstractVector{V},
    mask::AbstractMatrix{Bool},
    distances::AbstractArray{U,2},
    medium::MediumProperties
) where {U<:Real,V<:PointlikeCherenkovEmitter}

    n_sources = size(params, 2)
    n_targets = size(params, 3)

    if size(mask) != size(params)[2:3]
        error("Mask has length $(length(mask)), expected: $(n_sources * n_targets)")
    end

    T = MixtureModel{Univariate,Continuous,LocationScale{Float64,Continuous,Gamma{U}},Categorical{U,Vector{U}}}
    mixtures::Vector{T} = Vector{T}(undef, n_targets)
    mixtures_buf = Zygote.Buffer(mixtures)

    probs = params[3, :, :]

    probs[.!mask] .= 0

    c_ph = (c_vac_m_ns / get_refractive_index(800.0f0, medium))

    @inbounds for i in 1:n_targets
        if any(probs[:, i] .> 0)


            dists = [Gamma(params[1, j, i], params[2, j, i]) + sources[j].time + distances[j, i]/c_ph for j in 1:n_sources]
            mixtures_buf[i] = MixtureModel(dists, probs[:, i] ./ sum(probs[:, i]))
        end
    end
    mixtures = copy(mixtures_buf)
end

function shape_mixture_per_module(
    targets::AbstractVector{U},
    particle::Particle,
    medium::MediumProperties,
    precision::Real,
    model::Flux.Chain,
    trafos::AbstractVector{Symbol}) where {U<:PhotonTarget}

    predictions, sources, mask = evaluate_model(targets, particle, medium, precision, model, trafos)
    shape_mixture_per_module(predictions, sources, mask)
end


function poisson_dist_per_module(
    params::AbstractArray{U,3},
    sources::AbstractVector{V},
    mask::AbstractMatrix{Bool}) where {U<:Real,V<:PointlikeCherenkovEmitter}

    n_sources = size(params, 2)
    n_targets = size(params, 3)


    if size(mask) != size(params)[2:3]
        error("Mask has length $(length(mask)), expected: $(n_sources * n_targets)")
    end
        
    pred = [sum([params[3, i, j] * sources[i].photons for i in 1:n_sources if mask[i, j]]) for j in 1:n_targets]

    Poisson.(pred)
end

function poisson_dist_per_module(
    targets::AbstractVector{U},
    particle::Particle,
    medium::MediumProperties,
    precision::Real,
    model::Flux.Chain,
    trafos::AbstractVector{Symbol}
) where {U<:PhotonTarget}

    predictions, sources, mask = evaluate_model(targets, particle, medium, precision, model, trafos)
    poisson_dist_per_module(predictions, sources, mask)
end


function sample_event(
    poissons::AbstractVector{U},
    shapes::AbstractVector{V},
    sources::AbstractVector{W}) where {U<:Sampleable,V<:Sampleable,W<:PointlikeCherenkovEmitter}

    if size(poissons) != size(shapes) != size(sources)
        error("Vectors have to be of same size")
    end

    event = Vector{Vector{Float64}}(undef, size(poissons))
    for i in eachindex(poissons)
        n_ph = rand(poissons[i])
        if n_ph > 0
            event[i] = rand(shapes[i], n_ph)
        else
            event[i] = []
        end
    end

    event
end




end