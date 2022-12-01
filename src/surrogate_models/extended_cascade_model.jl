module ExtendedCascadeModel

using MLUtils
using Flux
using TensorBoardLogger
using SpecialFunctions
using Logging
using ProgressLogging
using Random

using ..RQSplineFlow: eval_transformed_normal_logpdf

export train_model, train_model!

"""
    RQNormFlowPoisson(K::Integer,
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
"""
struct RQNormFlowPoisson
    embedding::Chain
    K::Integer
    range_min::Float64
    range_max::Float64
end

# Make embedding parameters trainable
Flux.@functor RQNormFlowPoisson (embedding,)

function RQNormFlowPoisson(K::Integer,
    range_min::Number,
    range_max::Number,
    hidden_structure::AbstractVector{<:Integer};
    dropout=0.3,
    non_linearity=relu,
    split_final=true)

    model = []
    push!(model, Dense(24 => hidden_structure[1], non_linearity))
    push!(model, Dropout(dropout))
    for ix in 2:length(hidden_structure[2:end])
        push!(model, Dense(hidden_structure[ix-1] => hidden_structure[ix], non_linearity))
        push!(model, Dropout(dropout))
    end

    # 3 K + 1 for spline, 1 for shift, 1 for scale, 1 for log-expectation
    n_spline_params = 3 * K + 1
    if split_final
        final = Parallel(vcat,
                        Dense(hidden_structure[end] => n_spline_params + 2),
                        Dense(hidden_structure[end] => 1)
        )
    else
        final =  Dense(hidden_structure[end] => n_spline_params + 3)
    end

    push!(model, final)

    return RQNormFlowPoisson(Chain(model...), K, range_min, range_max)
end

"""
    (m::RQNormFlowPoisson)(x, cond)

Evaluate normalizing flow at values `x` with conditional values `cond`.

Returns logpdf and log-expectation
"""
function (m::RQNormFlowPoisson)(x, cond)
    params = m.embedding(Float64.(cond))
    spline_params = params[1:end-1, :]
    logpdf_eval = eval_transformed_normal_logpdf(x, spline_params, m.range_min, m.range_max)
    log_expec = params[end, :]

    return logpdf_eval, log_expec
end

"""
    rq_norm_flow_poisson_loss(x::NamedTuple, model::RQNormFlowPoisson)

Evaluate model and return sum of logpdfs of normalizing flow and poisson
"""
function log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlowPoisson)
    logpdf_eval, log_expec = model(x[:tres], x[:label])

    # poisson: log(exp(-lambda) * lambda^k)
    poiss_f = -exp.(log_expec) .+ x[:nhits] .* log_expec .- loggamma.(x[:nhits] .+ 1.0)

    return -sum(logpdf_eval), -sum(poiss_f)
end


Base.@kwdef struct RQNormFlowPoissonHParams
    K::Int64 = 10
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::Symbol = :relu
    seed::Int64 = 31338
    split_final = false
    rel_weight_poisson = 0.001
    use_l2_norm = false
end


function train_model(data, use_gpu=true; hyperparams...)

    hparams = RQNormFlowPoissonHParams(; hyperparams...)

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
        shuffle=true,
        rng=rng)


    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict(:relu => relu, :tanh => tanh)
    non_lin = non_lins[hparams.non_linearity]

    model = RQNormFlowPoisson(
        hparams.K, -5.0, 5.0, hidden_structure, dropout=hparams.dropout, non_linearity=non_lin)

    opt = Flux.Optimise.Adam(hparams.lr)

    logdir = joinpath(@__DIR__, "../../tensorboard_logs/RQNormFlowPoisson")
    lg = TBLogger(logdir)

    device = use_gpu ? gpu : cpu
    final_test_loss = train_model!(opt, train_loader, test_loader, model, hparams.epochs, lg, device, hparams.rel_weight_poisson, hparams.use_l2_norm)




    return model, final_test_loss
end


# Function to get dictionary of model parameters
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
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

function train_model!(opt, train, test, model, epochs, logger, device, rel_weight_poisson, use_l2_norm)
    model = model |> device
    pars = Flux.params(model)
    local train_loss_flow, train_loss_poisson
    local total_test_loss
    @progress for epoch in 1:epochs
        total_train_loss_flow = 0.0
        total_train_loss_poisson = 0.0

        Flux.trainmode!(model)
        for d in train
            d = d |> device
            gs = gradient(pars) do
                train_loss_flow, train_loss_poisson = log_likelihood_with_poisson(d, model)

                loss =  train_loss_flow + rel_weight_poisson * train_loss_poisson 
                if use_l2_norm
                    loss = loss +  sum(sqnorm, pars)
                end

                return loss
            end
            total_train_loss_flow += train_loss_flow / length(d[:tres])
            total_train_loss_poisson += train_loss_poisson / length(d[:tres])
            #=
            with_logger(logger) do
                @info "train" batch_loss=train_loss
            end
            =#
            Flux.update!(opt, pars, gs)
        end

        total_train_loss_flow /= length(train)
        total_train_loss_poisson /= length(train)
        total_train_loss = total_train_loss_flow +  total_train_loss_poisson

        Flux.testmode!(model)
        total_test_loss_flow = 0
        total_test_loss_poisson = 0
        for d in test
            d = d |> device
            test_loss_flow, test_loss_poisson = log_likelihood_with_poisson(d, model)
            total_test_loss_flow += test_loss_flow / length(d[:tres])
            total_test_loss_poisson += test_loss_poisson / length(d[:tres])
        end
        total_test_loss_flow /= length(test)
        total_test_loss_poisson /= length(test)
        total_test_loss = total_test_loss_flow + total_test_loss_poisson

        param_dict = Dict{String, Any}()
        fill_param_dict!(param_dict, model, "")
        
        
        with_logger(logger) do
            @info "loss" train_flow = total_train_loss_flow train_poisson = total_train_loss_poisson
            @info "loss" log_step_increment = 0 test_flow = total_test_loss_flow test_poisson = total_test_loss_poisson
            @info "loss" log_step_increment = 0 train_total = total_train_loss test_total = total_test_loss
            @info "model" params=param_dict log_step_increment=0

        end
        println("Epoch: $epoch, Train: $total_train_loss Test: $total_test_loss")

    end
    return total_test_loss
end
end
