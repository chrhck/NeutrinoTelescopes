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

device = cpu


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


fname = joinpath(@__DIR__, "../assets/photon_tables_extended_1.hd5")
fid = h5open(fname, "r")
g = fid["pmt_hits/dataset_600"]
attrs(g)

df = DataFrame(g[:, :], [:tres, :pmt_id])
pmt = 8
times = df[df.pmt_id.==pmt, :tres]

hppmt = combine(groupby(df, :pmt_id), nrow)

fig = Figure()
hist(fig[1, 1], times; bins=-10:0.5:20, normalization=:pdf)



struct RQNormFlow
    embedding::Chain
    sigma::Float64
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

bv = Bijectors.RationalQuadraticSpline(randn(2, K), randn(2, K), randn(2, K - 1), 5)
ibv = inverse(bv)

with_logabsdet_jacobian(ibv, [1.0, 1.0])

bv.widths

function test(a, b)
    @show typeof(a), typeof(b)
end

test.(bv.widths, ones(2, 1))


Bijectors.rqs_logabsdetjac.(bv.widths, bv.heights, bv.derivatives, [1, 1])


bv()



function (m::RQNormFlow)(x, cond)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.

    base = Normal()
    params = m.embedding(Float64.(cond))

    rqs_forward(b.widths, b.heights, b.derivatives, x)


    x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
    return logpdf(td.dist, x) + logjac



    res = @inbounds [
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

times = df[:, :tres]
pmt = onehotbatch(df[:, :pmt_id], 1:16)

data = DataLoader(
    (data=times |> device, label=pmt |> device),
    batchsize=50,
    shuffle=true)
rq_layer = RQNormFlow(K, B, [256, 256]) |> device

length(data)

loss(x) = -sum(rq_layer(x[:data], x[:label]))
loss(first(data))


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
