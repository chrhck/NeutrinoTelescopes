using NeutrinoTelescopes
using Flux
using CUDA
using Random
using StaticArrays
using BSON: @save, @load
using BSON
using CairoMakie
model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")

@load model_path model hparams opt tf_dict

pos = SA[0.0, 20.0, 0.0]
dir_theta = deg2rad(20)
dir_phi = deg2rad(50)
dir = sph_to_cart(dir_theta, dir_phi)

pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0

p = Particle(pos, dir, 0.0, 1E5, PEPlus)
target = MultiPMTDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32),
    UInt16(1)
)

wl_range = (300.0f0, 800.0f0)
medium = make_cascadia_medium_properties(0.99f0)
spectrum = CherenkovSpectrum(wl_range, 30, medium)

source = ExtendedCherenkovEmitter(p, medium, wl_range)

photons = PhotonPropSetup([p], [target], medium, spectrum)


input = calc_flow_inputs([p], [target], tf_dict)

times = -10:1:100


log_pdf, log_expec = model(repeat(times, size(input, 2)), repeat(input, inner=(1, length(times))), true)
log_pdf = reshape(log_pdf, length(times), size(input, 2),)

fig = Figure()
ga = fig[1, 1] = GridLayout(4, 4)
li = CartesianIndices((4, 4))
for i in 1:16
    row, col = divrem(i - 1, 4)
    lines(ga[col+1, row+1], times, exp.(log_pdf[:, i]))
end

fig
li[2]

lines()
lines(times, exp.(log_pdf[:, 2]))
lines(times, exp.(log_pdf[:, 3]))
lines(times, exp.(log_pdf[:, 4]))
