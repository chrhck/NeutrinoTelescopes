using NeutrinoTelescopes
using NeutrinoTelescopes.Modelling
using NeutrinoTelescopes.PMTFrontEnd
using NeutrinoTelescopes.Utils
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.Detection
using NeutrinoTelescopes.EventGeneration 
using Plots
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DataStructures
using StatsPlots


pmts_per_module = 16
pmt_cath_area_r = 75e-3 / 2  # m
pmt_area = π * pmt_cath_area_r^2


positions = make_detector_cube(5, 5, 10, 50.0, 100.0)
targets = make_targets(positions, pmts_per_module, pmt_area)
medium64 = make_cascadia_medium_properties(Float64)
data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
model = data[:model] |> gpu
output_trafos = [:log, :log, :neg_log_scale]






center = SA[0., 0., 0.]
vol = Cuboid(center, 500., 500., 500.)
pdist = CategoricalSetDistribution(OrderedSet([:EMinus, :EPlus]), [0.5, 0.5])
edist = Pareto(1, 1E4) + 1E4
ang_dist = UniformAngularDistribution()
inj = VolumeInjector(vol, edist, pdist, ang_dist)

particle = rand(inj)



model_params, sources, mask = evaluate_model(targets, particle, medium64, 0.5, model, output_trafos)

poissons = poisson_dist_per_module(model_params, sources, mask)
shapes = shape_mixture_per_module(model_params, sources, mask)

event = sample_event(poissons, shapes, sources)

xs = [pos[1] for pos in positions]
ys = [pos[2] for pos in positions]
zs = [pos[3] for pos in positions]

nph_ev = length.(event)
nonzero_mask = nph_ev .> 0
scatter(xs, ys, zs, marker=:dot, markercolor=:black)

scatter!(xs[nonzero_mask], ys[nonzero_mask], zs[nonzero_mask], marker_z = log10.(nph_ev[nonzero_mask]))



poissons


edist = Pareto(1, 1E5) + 1E5

plot(edist, xscale=:log10, yscale=:log10, xlim=(1E4, 1E8))


sum(length.(event))

mask