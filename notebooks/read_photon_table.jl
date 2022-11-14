using DataFrames
using HDF5
using Makie
using GLMakie
GLMakie.activate!()
using SphereSurfaceHistogram
using LinearAlgebra
using StatsBase
using DataStructures


fname = joinpath(@__DIR__, "../assets/photon_tables_extended_1.hd5")
fid = h5open(fname, "r")


binned_sph = SortedDict()
binned_tres = SortedDict()

for g in fid["photons"]

    df = DataFrame(
        g[:, :],
        [:tres, :pos_x, :pos_y, :pos_z, :total_weight]
    )

    df = df[df[:, :tres].>10, :]

    w = ProbabilityWeights(df[:, :total_weight])
    n_samps = 20000
    ixs = sample(1:nrow(df), w, n_samps)
    pos = Matrix(df[ixs, [:pos_x, :pos_y, :pos_z]])
    pos = pos ./ map(norm, eachrow(pos))
    binner = SSHBinner(500)
    @inbounds for p in eachrow(pos)
        push!(binner, p)
    end
    binned_sph[attrs(g)["distance"]] = binner
    binned_tres[attrs(g)["distance"]] = fit(StatsBase.Histogram, df[:, :tres], w, -50:3:100; closed=:left)
end

fig = Figure()
sl_x = Slider(fig[2, 1], range=1:length(binned_sph))
skeys = collect(keys(binned_sph))
plot_bins = lift(sl_x.value) do x
    binned_sph[skeys[x]]
end
histogram(fig[1, 1], plot_bins, colormap=:viridis)
Colorbar(fig[1, 2], label="Counts", colormap=:viridis, limits=(0, 1))
fig

"""
fig2 = Figure()
sl_x = Slider(fig2[2, 1], range=1:length(binned_tres))
plot_tres = lift(sl_x.value) do x
    plot_tres[skeys[x]].edges, plot_tres[skeys[x]].weights
end

barplot(plot_tres[1], plot_tres[2])
fig2
"""
