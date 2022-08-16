using Pkg
Pkg.activate(".")
using NeutrinoTelescopes.PhotonPropagationCuda 
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.Types
using NeutrinoTelescopes.Emission
using NeutrinoTelescopes.Spectral
using NeutrinoTelescopes.Detection
using StaticArrays
using CUDA

n_photons = Int64(1E5)
distance = 80f0
medium = Medium.make_cascadia_medium_properties(Float32)
source = PointlikeIsotropicEmitter(SA[0f0, 0f0, 0f0], 0f0, Int64(1E8), CherenkovSpectrum((300f0, 800f0), 50, medium))
target_radius = 0.21f0
n_pmts=16
pmt_area=Float32((75e-3 / 2)^2*π)

distance = Float32(distance)
target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)

threads = 640
blocks = 92
stack_len = Int32(1E6)

positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = initialize_photon_arrays(stack_len, blocks, Float32)
err_code = CuVector(zeros(Int32, 1))

spectrum_vals = CuTextureArray(source.spectrum.knots)
spectrum_texture = CuTexture(spectrum_vals; interpolation=CUDA.LinearInterpolation(), normalized_coordinates=true)

@CUDA.profile @cuda threads = threads blocks = blocks shmem = sizeof(Int64) cuda_propagate_photons!(
    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, stack_len, Int64(0),
    source, spectrum_texture, target.position, target.radius, Val(medium))


kernel = @cuda launch=false cuda_propagate_photons!(
    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, stack_len, Int64(0),
    source, spectrum_texture, target.position, target.radius, Val(medium))

CUDA.launch_configuration(kernel.fun; shmem=sizeof(Int64))