import NeutrinoTelescopes: PhotonPropagationCuda as ppcu
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.LightYield
using NeutrinoTelescopes.Emission
using NeutrinoTelescopes.Spectral
using NeutrinoTelescopes.Types
using NeutrinoTelescopes.Detection
using NeutrinoTelescopes.Utils
using StaticArrays
using Plots
using Traceur

medium = make_cascadia_medium_properties(Float32)


theta = deg2rad(90f0)
phi = deg2rad(-90f0)

dir = sph_to_cart(theta, phi)

particle = Particle(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        dir,
        0f0,
        Float32(1E5),
        PEMinus
)

source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))

source.photons / 1E9

ppcu.initialize_photon_state(source, medium)

dirs = [ppcu.initialize_photon_state(source, medium).direction for _ in 1:10000]

scatter([dir[1] for dir in dirs], [dir[2] for dir in dirs], [dir[3] for dir in dirs], markersize=1, alpha=0.2)


distance = 20f0
n_pmts=16
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0

target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)

results = ppcu.propagate_photons(source, target, medium, 512, 92, Int32(100000))






histogram(ppcu.process_output(Vector(results[5]), Vector(results[6])), bins=0:1:200)