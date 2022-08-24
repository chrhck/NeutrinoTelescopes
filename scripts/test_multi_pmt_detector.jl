using NeutrinoTelescopes
using Plots
using StaticArrays
using Random

distance = 80f0
medium = make_cascadia_medium_properties(Float32)
source = PointlikeIsotropicEmitter(SA[0f0, 0f0, 0f0], 0f0, Int64(1E8), CherenkovSpectrum((300f0, 800f0), 50, medium))
n_pmts=16
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0


make_pom_pmt_coordinates(Float32)

target = MultiPMTDetector(@SVector[0.0f0, 0.0f0, distance], target_radius, pmt_area, 
    make_pom_pmt_coordinates(Float32))


nph = 10000
rthetas = acos.(2 .* rand(nph) .- 1)
rphis = 2*π .* rand(nph)

positions = target.radius .* sph_to_cart.(rthetas, rphis) .+ [target.position]

scatter([p[1] for p in positions], [p[2] for p in positions], [p[3] for p in positions],
ms=0.1, alpha=0.5)

pmt_pos = [target.position .+ sph_to_cart(col...).* target.radius for col in eachcol(target.pmt_coordinates)]

scatter!([p[1] for p in pmt_pos], [p[2] for p in pmt_pos], [p[3] for p in pmt_pos],
ms=3, alpha=0.9, color=:red)

hit_pmts = check_pmt_hit.(positions, Ref(target))
hit_photons = positions[hit_pmts .!= 0]

scatter!([p[1] for p in hit_photons], [p[2] for p in hit_photons], [p[3] for p in hit_photons],
ms=0.5, alpha=0.9, color=:red)





sum(hit_pmts .!= 0)

pos = pmt_pos[end]

check_pmt_hit(pos, target)

test = SA[-0.37992823, -0.37992817, -0.8433914]
using LinearAlgebra
norm(test)

apply_rot(test, SA[0., 0., 1.], test)

rot_ez_fast(test, test)

b = SA[0., 0., 1.]
a = test



R * a



source = PointlikeIsotropicEmitter(SA[0f0, 0f0, 0f0], 0f0, Int64(ceil(nph)), CherenkovSpectrum((300f0, 800f0), 50, medium))


target.pmt_coordinates

propagate_photons(source, target, medium)