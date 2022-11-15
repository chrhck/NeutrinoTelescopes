using Bijectors
using HDF5
using DataFrames
using CairoMakie
using Flux
using Optim

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

base = Normal()
B = 5
K = 8

widths = randn(K)
heights = randn(K)
derivs = randn(K - 1)

model = Chain(
    Dense(16 => 512),
    Dense(16 => 3 * K - 1)
)



rq = Bijectors.RationalQuadraticSpline(widths, heights, derivs, B)
b = Bijectors.Scale(10) ∘ rq

td = transformed(base, b)
xs = -50:0.1:50
lines!(fig[1, 1], xs, pdf.(td, xs))
fig




Flux.params(widths, heights, derivs)
