using NeutrinoTelescopes
using Plots
using StaticArrays

function draw_greatcircle!(zenith, p, target)
    points = sph_to_cart.(Ref(deg2rad(zenith)), 0:0.01:2*π)
    points = apply_rot.(Ref([0., 0., 1.]), Ref([1., 0., 0.]), points)
    points = target.radius .* points .+ [target.position]


    p = scatter!(p, [p[1] for p in points], [p[2] for p in points], [p[3] for p in points],
    ms=0.1, alpha=1.0, color=:blue, label="")
    p
end

function draw_pmt_great_circles(p)

    p = draw_greatcircle!(90-57.5, p, target)
    p = draw_greatcircle!(90-25, p, target)

    p = draw_greatcircle!(90+57.5, p, target)
    p = draw_greatcircle!(90+25, p, target)
    p
end


distance = 50f0
medium = make_cascadia_medium_properties(Float32)
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = MultiPMTDetector(@SVector[distance, 0f0, 0f0], target_radius, pmt_area, 
    make_pom_pmt_coordinates(Float32))


nph = 10000
rthetas = acos.(2 .* rand(nph) .- 1)
rphis = 2*π .* rand(nph)

positions = target.radius .* sph_to_cart.(rthetas, rphis) .+ [target.position]
pmt_pos = [target.position .+ sph_to_cart(col...).* target.radius for col in eachcol(target.pmt_coordinates)]



#=p = scatter([p[1] for p in positions], [p[2] for p in positions], [p[3] for p in positions],
ms=0.1, alpha=0.5)
=#

p = scatter([p[1] for p in pmt_pos], [p[2] for p in pmt_pos], [p[3] for p in pmt_pos],
ms=3, alpha=0.9, color=:red)
draw_pmt_great_circles(p)

orientation = sph_to_cart(π/2, 0)

hit_pmts = check_pmt_hit.(positions, Ref(target), Ref(orientation))
hit_photons = positions[hit_pmts .!= 0]

scatter!([p[1] for p in hit_photons], [p[2] for p in hit_photons], [p[3] for p in hit_photons],
ms=0.5, alpha=0.9, color=:red)