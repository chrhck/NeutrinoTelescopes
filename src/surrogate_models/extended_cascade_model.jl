module ExtendedCascadeModel

using MLUtils
using Flux
using TensorBoardLogger
using SpecialFunctions
using Logging
using ProgressLogging

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
    non_linearity=relu)

    model = []
    push!(model, Dense(24 => hidden_structure[1], non_linearity))
    push!(model, Dropout(dropout))
    for ix in eachindex(hidden_structure[2:end])
        push!(model, Dense(hidden_structure[ix-1] => hidden_structure[ix], non_linearity))
        push!(model, Dropout(dropout))
    end
    # 3 K + 1 for spline, 1 for shift, 1 for scale, 1 for log-expectation
    n_spline_params = 3 * K + 1
    push!(model, Dense(hidden_structure[end] => n_spline_params + 3))

    return RQNormFlow(Chain(model...), K, range_min, range_max)
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
    K::Int64
    batch_size::Int64
    mlp_layers::Int64
    mlp_layer_size::Int64
    lr::Float64
    epochs::Int64
    dropout::Float64
    non_linearity::Symbol
end

function train_model(data, hyperparams...)

    hparams = RQNormFlowPoissonHParams(hyperparams...)

    train_data, test_data = splitobs(data, at=0.85)

    train_loader = DataLoader(
        train_data,
        batchsize=hparams.batch_size,
        shuffle=true,
        rng=rng)

    test_loader = DataLoader(
        test_data,
        batchsize=10000,
        shuffle=true,
        rng=rng)

    hidden_structure = fill(mlp_layer_size, mlp_layers)

    non_lins = Dict(:relu => relu, :tanh => tanh)
    non_lin = non_lins[hparams.non_linearity]

    model = RQNormFlowPoisson(
        hparams.K, -5.0, 5.0, hidden_structure, dropout=hparams.dropout, non_linearity=non_lin)

    opt = Flux.Optimise.Adam(hparams.lr)

    logdir = joinpath(@__DIR__, "../../tensorboard_logs/RQNormFlowPoisson")
    lg = TBLogger(logdir)

    train_model!(opt, train_loader, test_loader, model, hparams.epochs, lg)

end

function train_model!(opt, train, test, model, epochs, logger)
    pars = Flux.params(model)
    local train_loss_flow, train_loss_poisson
    @progress for epoch in 1:epochs
        total_train_loss_flow = 0.0
        total_train_loss_poisson = 0.0

        Flux.trainmode!(model)
        for d in train
            d = d |> device
            gs = gradient(pars) do
                train_loss_flow, train_loss_poisson = log_likelihood_with_poisson(d, model)
                return train_loss_flow + train_loss_poisson
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
        total_train_loss = total_train_loss_flow + total_train_loss_poisson

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

        with_logger(logger) do
            @info "loss/train" train_flow = total_train_loss_flow train_poisson = total_train_loss_poisson
            @info "loss/test" log_step_increment = 0 test_flow = total_test_loss_flow test_poisson = total_test_loss_poisson
            @info "loss/total" log_step_increment = 0 train_total = total_train_loss test_total = total_test_loss


        end
        println("Epoch: $epoch, Train: $total_train_loss Test: $total_test_loss")

    end
end
end
