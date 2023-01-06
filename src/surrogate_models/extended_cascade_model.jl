module ExtendedCascadeModel

using MLUtils
using Flux
using TensorBoardLogger
using SpecialFunctions
using Logging
using ProgressLogging
using Random
using EarlyStopping
using DataFrames
using CategoricalArrays
using HDF5
using StatsBase
using LinearAlgebra
using Flux.Optimise
using BSON: @save
using Base.Iterators
using PoissonRandom
using LogExpFunctions

using ..RQSplineFlow: eval_transformed_normal_logpdf, sample_flow
using ...Types
using ...Utils
using ...PhotonPropagation.Detection
using ...PhotonPropagation.PhotonPropagationCuda

export sample_cascade_event, single_cascade_likelihood, evaluate_model
export create_pmt_table, preproc_labels, read_pmt_hits, calc_flow_input, fit_trafo_pipeline, log_likelihood_with_poisson
export train_time_expectation_model, train_model!, RQNormFlowHParams, setup_time_expectation_model, setup_dataloaders
export Normalizer


abstract type ArrivalTimeSurrogate end

"""
    RQNormFlow(K::Integer,
                      range_min::Number,
                      range_max::Number,
                      hidden_structure::AbstractVector{<:Integer};
                      dropout::Real=0.3,
                      non_linearity=relu)

1-D rq-spline normalizing flow with expected counts prediction.

The rq-spline requires 3 * K + 1 parameters, where `K` is the number of knots. These are
parametrized by an embedding (MLP).

# Arguments
- K: Number of knots
- range_min:: Lower bound of the spline transformation
- range_max:: Upper bound of the spline transformation
- hidden_structure:  Number of nodes per MLP layer
- dropout: Dropout value (between 0 and 1) used in training (default=0.3)
- non_linearity: Non-linearity used in MLP (default=relu)
- add_log_expec: Also predict log-expectation
- split_final=false: Split the final layer into one for predicting the spline params and one for the log_expec
"""
struct RQNormFlow <: ArrivalTimeSurrogate
    embedding::Chain
    K::Integer
    range_min::Float64
    range_max::Float64
    has_log_expec::Bool
end

# Make embedding parameters trainable
Flux.@functor RQNormFlow (embedding,)

function RQNormFlow(K::Integer,
    range_min::Number,
    range_max::Number,
    hidden_structure::AbstractVector{<:Integer};
    dropout=0.3,
    non_linearity=relu,
    add_log_expec=false,
    split_final=false
)

    model = []
    push!(model, Dense(24 => hidden_structure[1], non_linearity))
    push!(model, Dropout(dropout))
    for ix in 2:length(hidden_structure[2:end])
        push!(model, Dense(hidden_structure[ix-1] => hidden_structure[ix], non_linearity))
        push!(model, Dropout(dropout))
    end

    # 3 K + 1 for spline, 1 for shift, 1 for scale, 1 for log-expectation
    n_spline_params = 3 * K + 1
    n_flow_params = n_spline_params + 2


    if add_log_expec && split_final
        final = Parallel(vcat,
            Dense(hidden_structure[end] => n_flow_params),
            Dense(hidden_structure[end] => 1)
        )
    elseif add_log_expec && !split_final
        #zero_init(out, in) = vcat(zeros(out-3, in), zeros(1, in), ones(1, in), fill(1/in, 1, in))
        final = Dense(hidden_structure[end] => n_flow_params + 1)
    else
        final = Dense(hidden_structure[end] => n_flow_params)
    end
    push!(model, final)

    return RQNormFlow(Chain(model...), K, range_min, range_max, add_log_expec)
end

"""
    (m::RQNormFlow)(x, cond)

Evaluate normalizing flow at values `x` with conditional values `cond`.

Returns logpdf and log-expectation
"""
function (m::RQNormFlow)(x, cond, pred_log_expec=false)
    params = m.embedding(Float64.(cond))

    @assert !pred_log_expec || (pred_log_expec && m.has_log_expec) "Requested to return log expectation, but model doesn't provide.
    "
    if pred_log_expec
        spline_params = params[1:end-1, :]
        logpdf_eval = eval_transformed_normal_logpdf(x, spline_params, m.range_min, m.range_max)
        log_expec = params[end, :]

        return logpdf_eval, log_expec
    else
        logpdf_eval = eval_transformed_normal_logpdf(x, params, m.range_min, m.range_max)
        return logpdf_eval
    end
end


"""
    log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)

Evaluate model and return sum of logpdfs of normalizing flow and poisson
"""
function log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)
    logpdf_eval, log_expec = model(x[:tres], x[:label], true)

    # poisson: log(exp(-lambda) * lambda^k)
    poiss_f = x[:nhits] .* log_expec .- exp.(log_expec) .- loggamma.(x[:nhits] .+ 1.0)

    # correct for overcounting the poisson factor
    poiss_f = poiss_f ./ x[:nhits]

    return -(sum(logpdf_eval) + sum(poiss_f)) / length(x[:tres])
end


"""
    log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)

Evaluate model and return sum of logpdfs of normalizing flow and poisson
"""
function log_likelihood(x::NamedTuple, model::RQNormFlow)
    logpdf_eval = model(x[:tres], x[:label], false)
    return -sum(logpdf_eval) / length(x[:tres])
end


Base.@kwdef struct RQNormFlowHParams
    K::Int64 = 10
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::Symbol = :relu
    seed::Int64 = 31338
    l2_norm_alpha = 0.0
end

function setup_time_expectation_model(hparams::RQNormFlowHParams)
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict(:relu => relu, :tanh => tanh)
    non_lin = non_lins[hparams.non_linearity]

    model = RQNormFlow(
        hparams.K, -20.0, 100.0, hidden_structure, dropout=hparams.dropout, non_linearity=non_lin,
        add_log_expec=true
    )
    return model
end

function setup_dataloaders(data, hparams::RQNormFlowHParams)
    train_data, test_data = splitobs(data, at=0.7)
    rng = Random.MersenneTwister(hparams.seed)

    train_loader = DataLoader(
        train_data,
        batchsize=hparams.batch_size,
        shuffle=true,
        rng=rng)

    test_loader = DataLoader(
        test_data,
        batchsize=50000,
        shuffle=false)

    return train_loader, test_loader
end


function train_time_expectation_model(data, use_gpu=true, use_early_stopping=true, checkpoint_path=nothing; hyperparams...)

    hparams = RQNormFlowHParams(; hyperparams...)

    model = setup_time_expectation_model(hparams)

    if hparams.l2_norm_alpha > 0
        opt = Optimiser(WeightDecay(hparams.l2_norm_alpha), Adam(hparams.lr))
    else
        opt = Adam(hparams.lr)
    end

    logdir = joinpath(@__DIR__, "../../tensorboard_logs/RQNormFlow")
    lg = TBLogger(logdir)

    train_loader, test_loader = setup_dataloaders(data, hparams)

    device = use_gpu ? gpu : cpu
    model, final_test_loss = train_model!(opt, train_loader, test_loader, model, log_likelihood_with_poisson, hparams, lg, device, use_early_stopping, checkpoint_path)

    return model, final_test_loss, hparams, opt
end


# Function to get dictionary of model parameters
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix * "layer_" * string(i) * "/" * string(layer) * "/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

sqnorm(x) = sum(abs2, x)

function train_model!(opt, train, test, model, loss_function, hparams, logger, device, use_early_stopping, checkpoint_path=nothing)
    model = model |> device
    pars = Flux.params(model)

    if use_early_stopping
        stopper = EarlyStopper(Warmup(Patience(5); n=3), InvalidValue(), NumberSinceBest(n=5), verbosity=1)
    else
        stopper = EarlyStopper(Never(), verbosity=1)
    end

    local loss
    local total_test_loss

    best_test = Inf

    @progress for epoch in 1:hparams.epochs

        Flux.trainmode!(model)

        total_train_loss = 0.0
        for d in train
            d = d |> device
            gs = gradient(pars) do
                loss = loss_function(d, model)

                return loss
            end
            total_train_loss += loss
            Flux.update!(opt, pars, gs)
        end


        total_train_loss /= length(train)

        Flux.testmode!(model)
        total_test_loss = 0
        for d in test
            d = d |> device
            total_test_loss += loss_function(d, model)
        end
        total_test_loss /= length(test)

        param_dict = Dict{String,Any}()
        fill_param_dict!(param_dict, model, "")


        with_logger(logger) do
            @info "loss" train = total_train_loss test = total_test_loss
            @info "model" params = param_dict log_step_increment = 0

        end
        println("Epoch: $epoch, Train: $total_train_loss Test: $total_test_loss")

        if !isnothing(checkpoint_path) && epoch > 5 && total_test_loss < best_test
            @save checkpoint_path*"_BEST.bson" model
            best_test = total_test_loss
        end

        done!(stopper, total_test_loss) && break

    end
    return model, total_test_loss
end


function create_pmt_table(grp, limit=true)
    hits = DataFrame(grp[:, :], [:tres, :pmt_id])
    for (k, v) in attrs(grp)
        hits[!, k] .= v
    end

    hits = DataFrames.transform!(groupby(hits, :pmt_id), nrow => :hits_per_pmt)

    if limit && (nrow(hits) > 200)
        hits = hits[1:200, :]
    end

    return hits
end

struct Normalizer{T}
    mean::T
    σ::T
end

Normalizer(x::AbstractVector) = Normalizer(mean(x), std(x))
#(norm::Normalizer)(x::AbstractVector) = promote(x .- norm.mean) ./ norm.σ
(norm::Normalizer)(x::Number) = (x - norm.mean) / norm.σ

function fit_normalizer!(x::AbstractVector)
    tf = Normalizer(x)
    x .= tf.(x)
    return x, tf
end


function dataframe_to_matrix(df)
    feature_matrix = Matrix{Float64}(undef, 9, nrow(df))
    feature_matrix[1, :] .= log.(df[:, :distance])
    feature_matrix[2, :] .= log.(df[:, :energy])

    feature_matrix[3:5, :] .= reduce(hcat, sph_to_cart.(df[:, :dir_theta], df[:, :dir_phi]))
    feature_matrix[6:8, :] .= reduce(hcat, sph_to_cart.(df[:, :pos_theta], df[:, :pos_phi]))
    feature_matrix[9, :] .= df[:, :pmt_id]

    return feature_matrix
end

function apply_feature_transform(m, tf_vec, n_pmt)

    lev = 1:n_pmt
    one_hot = (lev .== permutedims(m[9, :]))

    tf_matrix = mapreduce(
        t -> permutedims(t[2].(t[1])),
        vcat,
        zip(eachrow(m), tf_vec) 
    )

    return vcat(one_hot, tf_matrix)
end


function preproc_labels(df, n_pmt, tf_vec=nothing)

    feature_matrix = dataframe_to_matrix(df)

    if isnothing(tf_vec)
        tf_vec = Vector{Normalizer{Float64}}(undef, 8)
        for (row, ix) in zip(eachrow(feature_matrix), eachindex(tf_vec))
            tf = Normalizer(row)
            tf_vec[ix] = tf
        end
    end

    feature_matrix = apply_feature_transform(feature_matrix, tf_vec, n_pmt)

    return feature_matrix, tf_vec
end


function calc_flow_input(particle::Particle, target::PhotonTarget, tf_vec::AbstractVector)

    particle_pos = particle.position
    particle_dir = particle.direction
    particle_energy = particle.energy
    target_pos = target.position

    rel_pos = particle_pos .- target_pos
    dist = norm(rel_pos)
    normed_rel_pos = rel_pos ./ dist
 
    n_pmt = get_pmt_count(target)

    feature_matrix = repeat(
        [
            log(dist)
            log(particle_energy)
            particle_dir
            normed_rel_pos
        ],
        1, n_pmt)

    feature_matrix = vcat(feature_matrix, permutedims(1:n_pmt))

    return apply_feature_transform(feature_matrix, tf_vec, n_pmt)

end

function calc_flow_input(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec)
    
    res = mapreduce(
        t -> calc_flow_input(t[1], t[2], tf_vec),
        hcat,
        product(particles, targets))
    
    return res
end



function read_pmt_hits(fnames, nsel_frac=0.8, rng=nothing)

    all_hits = []
    for fname in fnames
        h5open(fname, "r") do fid
            if !isnothing(rng)
                datasets = shuffle(rng, keys(fid["pmt_hits"]))
            else
                datasets = keys(fid["pmt_hits"])
            end

            if nsel_frac == 1
                index_end = length(datasets)
            else
                index_end = Int(ceil(length(datasets) * nsel_frac))
            end


            for grpn in datasets[1:index_end]
                grp = fid["pmt_hits"][grpn]
                hits = create_pmt_table(grp)
                push!(all_hits, hits)
            end
        end
    end

    rnd_ixs = shuffle(rng, 1:length(all_hits))

    all_hits = all_hits[rnd_ixs]

    hits_df = reduce(vcat, all_hits)

    tres = hits_df[:, :tres]
    nhits = hits_df[:, :hits_per_pmt]
    cond_labels, tf_dict = preproc_labels(hits_df, 16)
    return tres, nhits, cond_labels, tf_dict
end

read_hdf(fname::String, nsel, rng) = read_hdf([fname], nsel, rng)




function poisson_logpmf(n, log_lambda)
    return n * log_lambda - exp(log_lambda) - loggamma(n + 1.0)
end


function sample_cascade_event(energy, dir_theta, dir_phi, position, time; targets, model, tf_vec, c_n, rng=nothing)
    
    dir = sph_to_cart(dir_theta, dir_phi)
    particle = Particle(position, dir, time, energy, PEMinus)
    input = calc_flow_input([particle], targets, tf_vec)    
    output = model.embedding(input)

    flow_params = output[1:end-1, :]
    log_expec = output[end, :]

    expec = exp.(log_expec)

    n_hits = pois_rand.(expec)
    mask = n_hits .> 0

    non_zero_hits = n_hits[mask]
    
    times = sample_flow(flow_params[:, mask], model.range_min, model.range_max, non_zero_hits, rng=rng)
    t_geos = [calc_tgeo(norm(particles[1].position .- targ.position) - targ.radius, c_n) for targ in targets]

    data = [length(ts) > 0 ? ts .- t_geos .- time : ts for ts in split_by(times, n_hits)]


    return data
end


function evaluate_model(particles, data, targets, model, tf_vec, c_n)
    n_pmt = get_pmt_count(eltype(targets))
    @assert length(targets)*n_pmt == length(data)

    t_geos = [calc_tgeo(norm(particles[1].position .- targ.position) - targ.radius, c_n) for targ in targets]

    input = calc_flow_input(particles, targets, tf_vec)
    
    output::Matrix{eltype(input)} = model.embedding(input)

    flow_params = output[1:end-1, :]
    log_expec_per_source = output[end, :] # one per source and pmt

    log_expec_per_source_rs = reshape(log_expec_per_source, length(targets)*n_pmt, length(particles))
    log_expec = sum(log_expec_per_source_rs, dims=2)[:, 1]
    rel_log_expec = log_expec_per_source_rs .- log_expec

    hits_per = length.(data)
    poiss = poisson_logpmf.(hits_per, log_expec)
    
    ix = LinearIndices((1:n_pmt*length(targets), eachindex(particles)))

    shape_llh_gen = ( 
        length(data[i]) > 0 ?
        LogExpFunctions.logsumexp(
            rel_log_expec[i, j] +
            sum(eval_transformed_normal_logpdf(
                data[i] .- t_geos - particles[j].time,
                repeat(flow_params[:, ix[i, j]], 1, hits_per[i]),
                model.range_min,
                model.range_max))
            for j in eachindex(particles)
        ) :
        0.
        for i in 1:n_pmt*length(targets)
    )
    
    return poiss, shape_llh_gen, log_expec
end

function single_cascade_likelihood(logenergy, dir_theta, dir_phi, position, time; data, targets, model, tf_vec)
    
    n_pmt = get_pmt_count(eltype(targets))

    @assert length(targets)*n_pmt == length(samples)
    dir = sph_to_cart(dir_theta, dir_phi)

    energy = 10^logenergy
    particles = [ Particle(position, dir, time, energy, PEMinus)]

    pois_llh, shape_llh, _ = evaluate_model(particles, data, targets, model, tf_vec)
    return sum(pois_llh) + sum(shape_llh)
end

end
