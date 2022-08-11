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

using Unitful
using PhysicalConstants.CODATA2018



data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
model = data[:model] |> gpu

output_trafos = [:log, :log, :neg_log_scale]

distance = 50f0
n_pmts=16
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)

targets = [target]

zenith_angle = 0f0
azimuth_angle = 0f0

pdir = sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle))

particle = Particle(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        pdir,
        0f0,
        Float32(1E4),
        PEMinus
)

medium = make_cascadia_medium_properties(Float32)
model_params, sources, mask, distances = evaluate_model(targets, particle, medium, 0.5f0, model, output_trafos)


poissons = poisson_dist_per_module(model_params, sources, mask)
shapes = shape_mixture_per_module(model_params, sources, mask, distances, medium)

event = sample_event(poissons, shapes, sources)

histogram(event)


prop_source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
prop_source2 = PointlikeIsotropicEmitter(particle.position, particle.time, prop_source.photons, CherenkovSpectrum((300f0, 800f0), 20, medium))

results = ppcu.propagate_photons(prop_source, target, medium, 512, 92, Int32(100000))
results = ppcu.propagate_photons(prop_source2, target, medium, 512, 92, Int32(100000))
positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = results

n_ph_sim = Vector(n_ph_sim)[1]
prop_source.photons


dist_travelled = process_output(Vector(dist_travelled), Vector(stack_idx))
wavelengths = process_output(Vector(wavelengths), Vector(stack_idx))
directions = process_output(Vector(directions), Vector(stack_idx))
times = process_output(Vector(times), Vector(stack_idx))


abs_weight = convert(Vector{Float64}, exp.(-dist_travelled ./ get_absorption_length.(wavelengths, Ref(medium))))

ref_ix = get_refractive_index.(wavelengths, Ref(medium))
c_vac = ustrip(u"m/ns", SpeedOfLightInVacuum)
# c_grp = get_group_velocity.(wavelengths, Ref(medium))


#photon_times = dist_travelled ./ c_grp

tgeo = (distance - target_radius) ./ (c_vac / get_refractive_index(800.0f0, medium))
tres = (times .- tgeo)

pmt_acc_weight = p_one_pmt_acc.(wavelengths)

log_dist, cos_obs_angle = source_to_input(sources[1], target)

#dir_weight = get_dir_reweight(thetas, acos(cos_obs_angle), ref_ix)
total_weight = abs_weight .* pmt_acc_weight

times

histogram(event[1] .- tgeo, normalize=true, bins=-50:50)
histogram!(tres, weight=total_weight,  bins=-50:50, normalize=true)

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
