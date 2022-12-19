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


using ..RQSplineFlow: eval_transformed_normal_logpdf
using ...Types
using ...Utils
using ...PhotonPropagation.Detection

export create_pmt_table, preproc_labels, read_pmt_hits, calc_flow_inputs, fit_trafo_pipeline, log_likelihood_with_poisson
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
    use_l2_norm = false
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


function train_time_expectation_model(data, use_gpu=true, use_early_stopping=true; hyperparams...)

    hparams = RQNormFlowHParams(; hyperparams...)

    model = setup_time_expectation_model(hparams)

    opt = Flux.Optimise.Adam(hparams.lr)

    logdir = joinpath(@__DIR__, "../../tensorboard_logs/RQNormFlow")
    lg = TBLogger(logdir)

    train_loader, test_loader = setup_dataloaders(data, hparams)

    device = use_gpu ? gpu : cpu
    model, final_test_loss = train_model!(opt, train_loader, test_loader, model, log_likelihood_with_poisson, hparams, lg, device, use_early_stopping)

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

function train_model!(opt, train, test, model, loss_function, hparams, logger, device, use_early_stopping)
    model = model |> device
    pars = Flux.params(model)

    if use_early_stopping
        stopper = EarlyStopper(Warmup(Patience(5); n=3), InvalidValue(), NumberSinceBest(n=5), verbosity=1)
    else
        stopper = EarlyStopper(Never(), verbosity=1)
    end

    local loss
    local total_test_loss

    @progress for epoch in 1:hparams.epochs

        Flux.trainmode!(model)

        total_train_loss = 0.0
        for d in train
            d = d |> device
            gs = gradient(pars) do
                loss = loss_function(d, model)
                if hparams.use_l2_norm
                    loss = loss + sum(sqnorm, pars)
                end

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


function preproc_labels(df, norm_dict=nothing)

    df[!, :log_distance] = log.(df[:, :distance])
    df[!, :log_energy] = log.(df[:, :energy])
    dir_cart = DataFrame(reduce(hcat, sph_to_cart.(df[:, :dir_theta], df[:, :dir_phi]))', [:dir_x, :dir_y, :dir_z])
    pos_cart = DataFrame(reduce(hcat, sph_to_cart.(df[:, :pos_theta], df[:, :pos_phi]))', [:pos_x, :pos_y, :pos_z])

    cond_labels = hcat(df[:, [:log_distance, :log_energy]], dir_cart, pos_cart)

    if isnothing(norm_dict)
        norm_dict = Dict{String,Normalizer}()
        for col in names(cond_labels)

            _, tf = fit_normalizer!(cond_labels[!, col])
            norm_dict[col] = tf
        end
    else
        for col in names(cond_labels)
            tf = norm_dict[col]

            cond_labels[!, col] .= tf(cond_labels[!, col])
        end
    end


    if "pmt_id" in names(df)
        lev = 1:16
        lev_names = Symbol.(Ref("pmt_"), Int.(lev))

        one_hot = DataFrame((lev .== permutedims(df[:, :pmt_id]))', lev_names)

        cond_labels = hcat(cond_labels, one_hot)
    end
    return cond_labels, norm_dict
end



function calc_flow_inputs(
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{T},
    tf_dict::Dict{String,<:Normalizer{NT}}
) where {T<:MultiPMTDetector,NT}

    n_pmt = get_pmt_count(T)
    li = LinearIndices((1:length(particles), 1:length(targets), 1:n_pmt))

    out = Matrix{Float64}(undef, 24, length(li))

    for i in eachindex(particles), j in eachindex(targets)

        p = particles[i]
        t = targets[j]

        log_distance = log(norm(p.position .- t.position))
        log_energy = log(p.energy)

        log_distance_tf = tf_dict["log_distance"](log_distance)
        log_energy_tf = tf_dict["log_energy"](log_energy)

        dir_theta, dir_phi = cart_to_sph(p.direction...)

        normed_pos = p.position ./ norm(p.position)
        pos_theta, pos_phi = cart_to_sph(normed_pos...)

        dir_x, dir_y, dir_z = sph_to_cart(dir_theta, dir_phi)
        pos_x, pos_y, pos_z = sph_to_cart(pos_theta, pos_phi)

        dir_x_tf::NT = tf_dict["dir_x"](dir_x)
        dir_y_tf::NT = tf_dict["dir_y"](dir_y)
        dir_z_tf::NT = tf_dict["dir_z"](dir_z)

        pos_x_tf = tf_dict["pos_x"](pos_x)
        pos_y_tf = tf_dict["pos_y"](pos_y)
        pos_z_tf = tf_dict["pos_z"](pos_z)

        for k in 1:16
            @show 1:16 .== k
            out[:, li[i, j, k]] .= vcat(1:16 .== k, [log_distance_tf, log_energy_tf, dir_x_tf, dir_y_tf, dir_z_tf, pos_x_tf, pos_y_tf, pos_z_tf])
        end
    end
    return out
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

    rnd_ixs = shuffle(1:length(all_hits))

    all_hits = all_hits[rnd_ixs]

    hits_df = reduce(vcat, all_hits)

    tres = hits_df[:, :tres]
    nhits = hits_df[:, :hits_per_pmt]
    cond_labels, tf_dict = preproc_labels(hits_df)
    return tres, nhits, cond_labels |> Matrix |> Adjoint, tf_dict
end

read_hdf(fname::String, nsel, rng) = read_hdf([fname], nsel, rng)


end
