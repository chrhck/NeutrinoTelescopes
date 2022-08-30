using NeutrinoTelescopes
using Plots
using StaticArrays
using Random
using LinearAlgebra
using DataFrames
using StatsPlots
using Distributions
using Parquet

distance = 20f0
medium = make_cascadia_medium_properties(Float32)
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = MultiPMTDetector(@SVector[distance, 0f0, 0f0], target_radius, pmt_area, 
    make_pom_pmt_coordinates(Float32))




nph = 10000
rthetas = acos.(2 .* rand(nph) .- 1)
rphis = 2*π .* rand(nph)

positions = target.radius .* sph_to_cart.(rthetas, rphis) .+ [target.position]


function draw_greatcircle!(zenith, p, target)
    points = sph_to_cart.(Ref(deg2rad(zenith)), 0:0.01:2*π)
    points = apply_rot.(Ref([0., 0., 1.]), Ref([1., 0., 0.]), points)
    points = target.radius .* points .+ [target.position]


    p = scatter!(p, [p[1] for p in points], [p[2] for p in points], [p[3] for p in points],
    ms=0.1, alpha=1.0, color=:blue)
    p
end

p = scatter([p[1] for p in positions], [p[2] for p in positions], [p[3] for p in positions],
ms=0.1, alpha=0.5)

p = plot()

p = draw_greatcircle!(90-57.5, p, target)
p = draw_greatcircle!(90-25, p, target)

p = draw_greatcircle!(90+57.5, p, target)
p = draw_greatcircle!(90+25, p, target)

pmt_pos = [target.position .+ sph_to_cart(col...).* target.radius for col in eachcol(target.pmt_coordinates)]

scatter!([p[1] for p in pmt_pos], [p[2] for p in pmt_pos], [p[3] for p in pmt_pos],
ms=3, alpha=0.9, color=:red)

orientation = sph_to_cart(π/2, 0)

hit_pmts = check_pmt_hit.(positions, Ref(target), Ref(orientation))
hit_photons = positions[hit_pmts .!= 0]

scatter!([p[1] for p in hit_photons], [p[2] for p in hit_photons], [p[3] for p in hit_photons],
ms=0.5, alpha=0.9, color=:red)


zenith_angle = 20f0
azimuth_angle = 10f0

pdir = sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle))

particle = Particle(
        @SVector[0.0f0, 0f0, 0.0f0],
        pdir,
        0f0,
        Float32(1E5),
        PEMinus
)

prop_source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))

res, nph_sim = propagate_photons(prop_source, target, medium)
res = make_hits_from_photons(res, source, target, medium)
groups = groupby(res, :pmt_id)
resampled_hits = combine(groups, [:time, :total_weight] => resample_simulation => :time)
resampled_hits

write_parquet("test_event.parquet", resampled_hits)


groupby(resampled_hits, :pmt_id)[2]


tres_tt = subtract_mean_tt(apply_tt(groups[5][:, :tres], STD_PMT_CONFIG.tt_dist), STD_PMT_CONFIG.tt_dist)
histogram(groups[5][:, :tres], weights=groups[5][:, :total_weight], xlim=(-10, 100), bins=-10:1:20)
histogram!(tres_tt, weights=groups[5][:, :total_weight], xlim=(-10, 100), bins=-10:1:20, alpha=0.8)
rs = resample_simulation(groups[5])


f

histogram!(rs, bins=-10:1:20, alpha=0.7)

ps =PulseSeries(rs, STD_PMT_CONFIG.spe_template, STD_PMT_CONFIG.pulse_model)
plot!(ps)


reco_pulses = make_reco_pulses(groups[5])
plot!(reco_pulses, xlim=(-10, 50))


refolded = PulseSeries(reco_pulses.times, reco_pulses.charges, STD_PMT_CONFIG.pulse_model)
plot!(refolded, xlim=(-10, 50))

@df combine(groups, nrow) scatter(:pmt_id, :nrow)



anim = @animate for zen in 0:20:180
    zenith_angle = Float32(zen)
    azimuth_angle = 10f0

    pdir = sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle))

    particle = Particle(
            @SVector[0.0f0, 0f0, 0.0f0],
            pdir,
            0f0,
            Float32(1E5),
            PEMinus
    )

    prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))

    res, nph_sim = propagate_photons(prop_source_ext, target, medium)
    res = make_hits_from_photons(res, source, target, medium)
    groups = groupby(res, :pmt_id)

    plots = []
    for i in 1:get_pmt_count(target)

        p = plot()
        if haskey(groups, i)
            group = groups[i]
            #p = histogram(group[:, :tres], weights=group[:, :total_weight], label="PMT: $i", bins=-10:50)
            reco_pulses = make_reco_pulses(group)
            if length(reco_pulses) > 0
                p = plot!(p, reco_pulses, xlim=(-50, 200))
            end
        end
        push!(plots, p)
    end
    plot(plots..., layout=@layout(grid(4, 4)), size=(1600, 1200), title="Zenith: $zenith_angle")
end

gif(anim, "test.gif", fps=2)