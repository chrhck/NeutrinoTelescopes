using NeutrinoTelescopes
using Plots
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DataFrames


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
        @SVector[0.0f0, 0f0, 0.0f0],
        pdir,
        0f0,
        Float32(1E5),
        PEMinus
)

medium = make_cascadia_medium_properties(Float32)

prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))

particle_to_elongated_lightsource(particle, (0f0, 20f0), 0.5f0, medium, (300f0, 800f0))


prop_source_che = PointlikeCherenkovEmitter(particle, medium, (300f0, 800f0))

results_che, nph_sim_che = propagate_source(prop_source_che, distance, medium)
results_ext, nph_sim_ext = propagate_source(prop_source_ext, distance, medium)