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

using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_increment, set_step!, set_step_increment!

using ..Emission
using ..Detection
using ..Medium
using ..PhotonPropagationCuda
using ..LightYield


export get_dir_reweight, fit_photon_dist, make_photon_fits
export Hyperparams, get_data
export splitdf, read_from_parquet
export loss_all, train_mlp
export source_to_input
export apply_transformation, reverse_transformation
export poisson_dist_per_module, shape_mixture_per_module, evaluate_model, sample_event

"""
    get_dir_reweight(thetas::AbstractVector{T}, obs_angle::T, ref_ixs::AbstractVector{T})

Calculate reweighting factor for photons from isotropic (4pi) emission to 
Cherenkov angular emission.

`thetas` are the photon zenith angles (relative to e_z)
`obs_angle` is the observation angle (angle of the line of sight between receiver 
and emitter and the Cherenkov emitter direction)
`ref_ixs` are the refractive indices for each photon
"""
function get_dir_reweight(thetas::AbstractVector{T}, obs_angle::Real, ref_ixs::AbstractVector{T}) where {T<:Real}
    norm = cherenkov_ang_dist_int.(ref_ixs) .* 2
    cherenkov_ang_dist.(cos.(thetas .- obs_angle), ref_ixs) ./ norm
end


function fit_photon_dist(obs_angles, obs_photon_df, n_ph_gen)
    df = DataFrame(
        obs_angle=Float64[],
        fit_alpha=Float64[],
        fit_theta=Float64[],
        det_fraction=Float64[])



    ph_thetas = obs_photon_df[:, :initial_theta]
    ph_ref_ix = obs_photon_df[:, :ref_ix]
    ph_abs_weight = obs_photon_df[:, :abs_weight]
    ph_tres = obs_photon_df[:, :tres]

    pmt_acc_weight = p_one_pmt_acc.(obs_photon_df[:, :wavelength])

    for obs_angle in obs_angles
        dir_weight = get_dir_reweight(ph_thetas, obs_angle, ph_ref_ix)
        total_weight = dir_weight .* ph_abs_weight .* pmt_acc_weight

        mask = ph_tres .>= 0

        dfit = fit_mle(Gamma, ph_tres[mask], total_weight[mask])
        push!(df, (obs_angle, dfit.α, dfit.θ, sum(total_weight) / n_ph_gen))
    end

    df
end

"""
    make_photon_fits(n_photons_per_dist::Int64, n_distances::Integer, n_angles::Integer)

Convenience function for propagating photons and fitting the arrival time distributions.
"""
function make_photon_fits(n_photons_per_dist::Int64, n_distances::Integer, n_angles::Integer, max_dist::Float32=300.0f0)

    s = SobolSeq([0.0], [pi])
    medium = make_cascadia_medium_properties(Float32)

    s2 = SobolSeq([0.0f0], [Float32(log10(max_dist))])
    distances = 10 .^ reduce(hcat, next!(s2) for i in 1:n_distances)

    results = Vector{DataFrame}(undef, n_distances)

    @progress name = "Propagating photons" for (i, dist) in enumerate(distances)
        results[i] = propagate_distance(dist, medium, n_photons_per_dist)

    end

    obs_angles = reduce(hcat, next!(s) for i in 1:n_angles)

    results_fit = Vector{DataFrame}(undef, n_distances)

    @progress name = "Propagating photons" for (i, dist) in enumerate(distances)
        results_fit[i] = fit_photon_dist(obs_angles, results[i], n_photons_per_dist)
    end

    vcat(results_fit..., source=:distance => vec(distances))

end

Base.@kwdef mutable struct Hyperparams
    data_file::String
    batch_size::Int64
    learning_rate::Float64
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
    target_names = [:log_fit_alpha, :log_fit_theta, :log_det_fraction_scaled]

    df_train, df_test = splitdf(results_df, 0.8)

    features_train = Matrix{Float32}(df_train[:, feature_names])'
    targets_train = Matrix{Float32}(df_train[:, target_names])'
    features_test = Matrix{Float32}(df_test[:, feature_names])'
    targets_test = Matrix{Float32}(df_test[:, target_names])'

    return (features_train, targets_train, features_test, targets_test)
end

function get_data(args::Hyperparams)

    trafos = Dict(
        (:det_fraction, :log_det_fraction) => :neg_log,
        (:det_fraction, :log_det_fraction_scaled) => :neg_log_scale,
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
    optimiser = ADAM(args.learning_rate)

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

    for epoch in 1:args.epochs
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

function source_to_input(source::CherenkovSegment, target::PhotonTarget)
    em_rec_vec = source.position .- target.position
    distance = norm(em_rec_vec)
    em_rec_vec = em_rec_vec ./ distance
    cos_obs_angle = dot(em_rec_vec, source.direction)

    return Float32(log10(distance)), Float32(cos_obs_angle)

end

function source_to_input(sources::AbstractVector{<:CherenkovSegment}, targets::AbstractVector{<:PhotonTarget})

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
    particle::LightYield.Particle,
    medium::Medium.MediumProperties,
    precision::T,
    model::Flux.Chain,
    trafos::AbstractVector{Symbol}
) where {U<:Detection.PhotonTarget,T<:Real}

    sources = LightYield.particle_to_elongated_lightsource(
        particle,
        (0.0, 20.0),
        precision,
        medium,
        (300.0, 800.0),
    )

    inputs = source_to_input(sources, targets)
    predictions::Matrix{Float32} = cpu(model(gpu(inputs)))
    Modelling.transform_model_output!(predictions, trafos)
    predictions_rshp = reshape(predictions, (3, size(sources, 1), size(targets, 1)))

    return predictions_rshp, sources
end

function shape_mixture_per_module(
    params::AbstractArray{U,3},
    sources::AbstractVector{V}
) where {U<:Real,V<:LightYield.CherenkovSegment}

    n_sources = size(params, 2)
    n_targets = size(params, 3)

    T = MixtureModel{Univariate,Continuous,LocationScale{Float64,Continuous,Gamma{U}},Categorical{U,Vector{U}}}
    mixtures::Vector{T} = Vector{T}(undef, n_targets)
    mixtures_buf = Zygote.Buffer(mixtures)

    @inbounds for i in 1:n_targets
        dists = [Gamma(params[1, j, i], params[2, j, i]) + sources[j].time for j in 1:n_sources]
        probs = params[3, :, i] ./ sum(params[3, :, i])
        mixtures_buf[i] = MixtureModel(dists, probs)
    end
    mixtures = copy(mixtures_buf)
end

function shape_mixture_per_module(
    targets::AbstractVector{U},
    particle::LightYield.Particle,
    medium::Medium.MediumProperties,
    precision::Real,
    model::Flux.Chain,
    trafos::AbstractVector{Symbol}) where {U<:Detection.PhotonTarget}

    predictions, sources = evaluate_model(targets, particle, medium, precision, model, trafos)
    shape_mixture_per_module(predictions, sources)
end


function poisson_dist_per_module(
    params::AbstractArray{U,3},
    sources::AbstractVector{V}) where {U<:Real,V<:CherenkovSegment}

    n_sources = size(params, 2)
    n_targets = size(params, 3)

    pred = [sum([params[3, i, j] * sources[i].photons for i in 1:n_sources]) for j in 1:n_targets]


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

    predictions, sources = evaluate_model(targets, particle, medium, precision, model, trafos)
    poisson_dist_per_module(predictions, sources)
end


function sample_event(
    poissons::AbstractVector{U},
    shapes::AbstractVector{V},
    sources::AbstractVector{W}) where {U<:Sampleable,V<:Sampleable,W<:CherenkovSegment}

    if size(poissons) != size(shapes) != size(sources)
        error("Vectors have to be of same size")
    end

    event = Vector{Vector{Float64}}(undef, size(poissons))
    for i in eachindex(poissons)
        n_ph = rand(poissons[i])
        event[i] = rand(shapes[i], n_ph)
    end

    event
end




end