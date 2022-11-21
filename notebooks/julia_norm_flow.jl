using Bijectors
using HDF5
using DataFrames
using CairoMakie
using Flux
using Optim
using Base.Iterators
using OneHotArrays
using TableTransforms
using MLUtils
using NNlib
using ProgressLogging
using CUDA
using BenchmarkTools
using Random
using CategoricalArrays

addevice = cpu


function scale_interval(x::Number, old::NTuple{2,<:Number}, new::NTuple{2,<:Number})

    a = (new[2] - new[1]) / (old[2] - old[1])
    b = new[2] - a * old[2]

    return a * x + b
end

function scale_interval(
    x::AbstractVector{<:Number},
    old::NTuple{2,<:Number},
    new::NTuple{2,<:Number})

    return scale_interval.(x, Ref(old), Ref(new))
end


fname = joinpath(@__DIR__, "../assets/photon_table_extended_2.hd5")
fid = h5open(fname, "r")



datasets = shuffle(keys(fid["pmt_hits"]))
nsel = 500


all_hits = []
for grpn in datasets[1:nsel]
    grp = fid["pmt_hits"][grpn]
    hits = DataFrame(grp[:, :], [:tres, :pmt_id])
    labels = attrs(grp)
    for (k, v) in attrs(grp)
        hits[!, k] .= v
    end
    push!(all_hits, hits)
end

data_df = vcat(all_hits...)
data_df[!, :pmt_id] = categorical(data_df[:, :pmt_id], levels=1:16)
data_df = data_df |> OneHot(:pmt_id)

data_df[!, :log_distance] = log.(data_df[:, :distance])
data_df[!, :log_energy] = log.(data_df[:, :energy])
data_df


struct RQNormFlow
    embedding::Chain
    K::Integer
    B::Integer
end

Flux.@functor RQNormFlow (embedding,)

function RQNormFlow(K, B, hidden_structure)

    model = []
    push!(model, Dense(16 => hidden_structure[1], relu))
    for ix in 2:length(hidden_structure)
        push!(model, Dense(hidden_structure[ix-1] => hidden_structure[ix], relu))
    end
    push!(model, Dense(hidden_structure[end] => 3 * K))

    return RQNormFlow(Chain(model...), K, B)
end


function (m::RQNormFlow)(x, cond)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.

    base = Normal()
    params = m.embedding(Float64.(cond))
    
    res = CUDA.@allowscalar @inbounds [
        logpdf(
            transformed(
                base,
                Bijectors.Scale(sigmoid(p[end]) * 50) ∘ Bijectors.RationalQuadraticSpline(
                    p[1:m.K],
                    p[m.K+1:2*m.K],
                    p[2*m.K+1:end-1],
                    m.B
                )),
            ax)
        for (p, ax) in zip(eachcol(params), x)
    ]

    return res

end

function train_model!(loss, data, pars, epochs)
    local train_loss
    @progress for epoch in 1:epochs
        total_loss = 0
        for d in data
            gs = gradient(pars) do
                train_loss = loss(d)
                return train_loss
            end
            total_loss += train_loss
            Flux.update!(opt, pars, gs)
        end
        total_loss /= length(data)
        println("Epoch: $epoch, Loss: $total_loss")

    end
end


B = 5
K = 10

device = cpu

feat_labels = []

data = DataLoader(
    data=data_df[:, :tres],
    labe
)

data = DataLoader(
    (data=times |> device, label=pmt |> device),
    batchsize=50,
    shuffle=true)
rq_layer = RQNormFlow(K, B, [256, 256]) |> device


loss(x) = -sum(rq_layer(x[:data], x[:label]))
@benchmark $loss($first($data))


pars = Flux.params(rq_layer)
opt = Flux.Optimise.Adam(0.001)
epochs = 100


train_model!(loss, data, pars, epochs)

pmt_plot = 11
t_plot = -5:0.1:20
l_plot = onehotbatch(fill(pmt_plot, length(t_plot)), 1:16)
fig = Figure()
lines(fig[1, 1], t_plot, exp.(rq_layer(t_plot, l_plot)))
hist!(fig[1, 1], df[df.pmt_id.==pmt_plot, :tres], bins=-5:0.5:50, normalization=:pdf)

fig


first(data)

rq_layer(randn(100), randn((16, 100)))


rq = Bijectors.RationalQuadraticSpline(widths, heights, derivs, B)
b = Bijectors.Scale(10) ∘ rq

td = transformed(base, b)
xs = -50:0.1:50
lines!(fig[1, 1], xs, pdf.(td, xs))
fig




Flux.params(widths, heights, derivs)
