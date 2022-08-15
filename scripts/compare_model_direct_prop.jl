using NeutrinoTelescopes
using NeutrinoTelescopes.Modelling
using NeutrinoTelescopes.PMTFrontEnd
using NeutrinoTelescopes.Utils
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.Detection
using NeutrinoTelescopes.Types
using NeutrinoTelescopes.LightYield
using NeutrinoTelescopes.Spectral
import NeutrinoTelescopes: PhotonPropagationCuda as ppcu
using NeutrinoTelescopes.Emission
using Plots
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DSP
using DataFrames

using LinearAlgebra

using Unitful
using PhysicalConstants.CODATA2018



data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
model = data[:model] |> gpu

output_trafos = [:log, :log, :neg_log_scale]

distance = 150f0
n_pmts=16
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)

targets = [target]

zenith_angle = 0f0
azimuth_angle = 0f0

pdir = sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle))

particle = Particle(
        @SVector[0.0f0, 0f0, 0.0f0],
        pdir,
        0f0,
        Float32(1E5),
        PEMinus
)

medium = make_cascadia_medium_properties(Float32)
model_params, sources, mask, distances = evaluate_model(targets, particle, medium, 0.5f0, model, output_trafos)


poissons = poisson_dist_per_module(model_params, sources, mask)
shapes = shape_mixture_per_module(model_params, sources, mask, distances, medium)

event = sample_event(poissons, shapes, sources)

histogram(event)

#=function propagate_source(source, target)
    results = ppcu.propagate_photons(source, target, medium, 512, 92, Int32(100000))
    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = results

    n_ph_sim = Vector(n_ph_sim)[1]

    dist_travelled = ppcu.process_output(Vector(dist_travelled), Vector(stack_idx))
    wavelengths = ppcu.process_output(Vector(wavelengths), Vector(stack_idx))
    directions = ppcu.process_output(Vector(directions), Vector(stack_idx))
    times = ppcu.process_output(Vector(times), Vector(stack_idx))

    abs_weight = convert(Vector{Float64}, exp.(-dist_travelled ./ get_absorption_length.(wavelengths, Ref(medium))))

    ref_ix = get_refractive_index.(wavelengths, Ref(medium))
    thetas = map(dir -> acos(dir[3]), directions)
    #c_vac = ustrip(u"m/ns", SpeedOfLightInVacuum)
    # c_grp = get_group_velocity.(wavelengths, Ref(medium))


    #photon_times = dist_travelled ./ c_grp

    #tgeo = (distance - target_radius) ./ (c_vac / get_refractive_index(800.0f0, medium))
    #tres = (times .- tgeo)

    pmt_acc_weight = p_one_pmt_acc.(wavelengths)

    total_weight = abs_weight .* pmt_acc_weight

    return DataFrame(:times => times, :total_weight => total_weight, :thetas=>thetas, :directions=>directions, :ref_ix=>ref_ix)
end
=#

prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
prop_source_iso = PointlikeIsotropicEmitter(particle.position, particle.time, prop_source_ext.photons, CherenkovSpectrum((300f0, 800f0), 20, medium))
prop_source_che = PointlikeCherenkovEmitter(particle, medium, (300f0, 800f0))


results_iso = ppcu.propagate_source(prop_source_iso, distance, medium)
results_che = ppcu.propagate_source(prop_source_che, distance, medium)
results_ext = ppcu.propagate_source(prop_source_ext, distance, medium)

dir_weight =  get_dir_reweight.(results_iso[:, :directions], Ref(prop_source_che.direction), results_iso[:, :ref_ix])


histogram(results_che[1][:, :tres], bins=-10:200)


scatter(cos.(results_iso[1:500000, :thetas]), dir_weight[1:500000], alpha=1)

histogram(cos.(results_che[:, :thetas]), weights=results_che[:, :total_weight], bins=-0:0.01:1, yscale=:log10, normalize=true)
histogram!(cos.(results_iso[:, :thetas]), weights=results_iso[:, :total_weight].* dir_weight, bins=-0:0.01:1, yscale=:log10, normalize=true)

histogram(results_che[:, :times], weights=results_che[:, :total_weight], bins=220:260, normalize=true)
histogram!(results_iso[:, :times], weights=results_iso[:, :total_weight] .* dir_weight, bins=220:260, normalize=true)

histogram(event[1], bins=200:5:400)
histogram!(results_ext[:, :times], weights=results_ext[:, :total_weight],  bins=200:5:400)





@show cos(results_iso[1, :directions][3] - dot(src_targ, prop_source.direction))

results_iso[1, :directions]

dir_weight_new = get_dir_reweight_new.(results_iso[:, :directions], Ref(src_targ), results_iso[:, :ref_ix])

results_iso[!, :norm_weights] = 
results_iso[!, :norm_weights] ./= sum(results_iso[!, :norm_weights])

results_che[!, :norm_weights] = results_che[:, :total_weight] / sum(results_che[:, :total_weight])


all(isfinite.(dir_weight_new))

histogram(cos.(results_che[:, :thetas]), weights=results_che[:, :total_weight],
 bins = -1:0.1:1,   normalize=true)
histogram!(cos.(results_iso[:, :thetas]), weights=results_iso[:, :total_weight].*results_iso[:, :dir_weight],
 bins = -1:0.1:1, legend=:topleft,  normalize=true)
histogram!(cos.(results_iso[:, :thetas]), weights=results_iso[:, :total_weight].*dir_weight_new,
 bins = -1:0.1:1, legend=:topleft, normalize=true)


histogram(results_iso[:, :times], weights=results_iso[:, :total_weight].*results_iso[:, :dir_weight],
bins = 280:1:360)
histogram!(results_che[:, :times], weights=results_che[:, :total_weight],
 bins = 280:1:360)
histogram!(results_iso[:, :times], weights=results_iso[:, :total_weight].*dir_weight_new,
bins = 280:1:360)

times


test = [0, 0, 0, 10, 10, 10]
weights = [10, 10, 10, 1, 1, 1]

histogram(test, weights=weights, bins=0:11)



sources

minimum(tres)

plot([source.time for source in sources], [source.photons for source in sources])


long_param = prop_source.long_param

scale = (1 / long_param.b)
shape = (long_param.a)

medium64 = make_cascadia_medium_properties(Float64)
long_param = LongitudinalParameterisation(1E5, medium64, get_longitudinal_params(PEMinus))


states = [ ppcu.initialize_photon_state(prop_source, medium) for _ in 1:100000]

histogram([source.time for source in states], normalize=true)
plot!([source.time for source in sources], [source.photons for source in sources] ./ 4E9)


histogram([source.position[3] for source in states], normalize=true)
plot!([source.position[3] for source in sources], [source.photons for source in sources] ./ 1E9)

track_dir = sample_cherenkov_track_direction(Float32)
dir_rot = ppcu.rotate_to_axis(prop_source.direction, track_dir)

prop_source.direction

sources[1].position
sources[1].time

sources[1].position[3] / sources[1].time 
states[1].position[3] / states[1].time 
