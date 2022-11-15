using Bijectors
using HDF5
using DataFrames
using CairoMakie
using Flux
using Optim
using Base.Iterators

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


B = 5
K = 8

widths = randn(K)
heights = randn(K)
derivs = randn(K - 1)

model = Chain(
    Dense(16 => 512),
    Dense(512 => 3 * K - 1)
)

struct RQNormFlow
    embedding::Chain
    K::Integer
    B::Integer
end

function (m::RQNormFlow)(x, cond)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
  
    base = Normal()

    params = m.embedding(cond)
   
    res = [
        logpdf(transformed(base, Bijectors.RationalQuadraticSpline(
            p[1:m.K],
            p[m.K+1:2*m.K],
            p[2*m.K+1:end],
            m.B
            )), ax)
        for (p, ax) in zip(eachcol(params), x)
    ]

    return res

end


data = Matrix(df)

data_b = [data[r, :]  for r in partition(1:nrow(df), 100)]


Flux.@functor RQNormFlow (embedding,)
rq_layer = RQNormFlow(model, K, 5)
loss(x) = -sum(rq_layer(x))

pars = Flux.params(rq_layer)
opt = Flux.Optimise.Adam()
Flux.train!(loss, pars, data, opt)


rq_layer(randn(100), randn((16, 100)))

a = collect(1:3*K-1)

a[2*K+1:end]


rq = Bijectors.RationalQuadraticSpline(widths, heights, derivs, B)
b = Bijectors.Scale(10) ∘ rq

td = transformed(base, b)
xs = -50:0.1:50
lines!(fig[1, 1], xs, pdf.(td, xs))
fig




Flux.params(widths, heights, derivs)
