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


center = SA[0., 0., 0.]

vol = Cuboid(center, 1000., 1000., 1000.)
pdist = CategoricalSetDistribution(OrderedSet([:EMinus, :EPlus]), [0.5, 0.5])
edist = Pareto(1, 100) + 100

ang_dist = UniformAngularDistribution()

inj = VolumeInjector(vol, edist, pdist, ang_dist)
particle = rand(inj)

positions = make_detector_cube(5, 5, 10, 50.0, 100.0)
targets = make_targets(positions)

medium64 = make_cascadia_medium_properties(Float64)

data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
model = data[:model] |> gpu

output_trafos = [:log, :log, :neg_log_scale]

model_params, sources = evaluate_model(targets, particle, medium64, 0.5, model, output_trafos)


poissons = poisson_dist_per_module(model_params, sources)
shapes = shape_mixture_per_module(model_params, sources)
