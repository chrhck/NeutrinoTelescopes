import NeutrinoTelescopes: PhotonPropagationCuda as ppcu
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.LightYield
using NeutrinoTelescopes.Emission
using NeutrinoTelescopes.Spectral
using NeutrinoTelescopes.Types
using NeutrinoTelescopes.Detection
using StaticArrays
using Plots
using Traceur

medium = make_cascadia_medium_properties(Float32)

source = PhotonSource(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        @SVector[0.0f0, 0.0f0, 1.0f0],
        0.0f0,
        Int64(1E9),
        CherenkovSpectrum((300.0f0, 800.0f0), 20, medium),
        AngularEmissionProfile{:CherenkovEmission,Float32}(),
        Float32(1E6),
        PEMinus)

ppcu.initialize_photon_state(source, medium)

distance = 20f0
n_pmts=16
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0

target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)

results = ppcu.propagate_photons(source, target, medium, 512, 92, Int32(100000))


histogram(ppcu.process_output(Vector(results[5]), Vector(results[6])), bins=0:1:100)