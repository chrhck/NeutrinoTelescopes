using NeutrinoTelescopes
using Plots
using StaticArrays
using Random
using LinearAlgebra
using DataFrames
using StatsPlots
using Distributions
using Parquet
using ProgressLogging
using Formatting
using JSON
using Arrow

medium = make_cascadia_medium_properties(Float32)
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
distance = 10f0

target = MultiPMTDetector(
    @SVector[distance, 0f0, 0f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32)
    )




outdir = joinpath(@__DIR__, "../assets/")

dfs = []
sim_params = []

@progress "Photon sims" for i in 1:10
    dir_costheta = rand(Uniform(-1, 1))
    dir_phi = rand(Uniform(0, 2*π))
    direction::SVector{3, Float32} = sph_to_cart(acos(dir_costheta), dir_phi)

    particle = Particle(
            @SVector[0.0f0, 0f0, 0.0f0],
            direction,
            0f0,
            Float32(1E5),
            PEMinus
    )

    n_photons::Int64 = 1E11
    prop_source = PointlikeCherenkovEmitter(particle, medium, n_photons, (300f0, 800f0))
    photons = propagate_photons(prop_source, target, medium)

    Arrow.write(
        joinpath(outdir, format("photons_{:d}", i)),
        photons;
        metadata=["target" => json(target), "source" => json(prop_source)])
    #=
    @progress "Resampling" for j in 1:10
        coszen_orient = rand(Uniform(-1, 1))
        phi_orient = rand(Uniform(0, 2*π))

        orientation = sph_to_cart(acos(coszen_orient), phi_orient,)
        hits = make_hits_from_photons(photons, prop_source, target, medium, orientation)

        push!(dfs, hits)
        push!(sim_params, (distance, dir_costheta, dir_phi, coszen_orient, phi_orient))
    end
    =#
end



groups = groupby(hits, :pmt_id)

expec_per_pmt = combine(groups, :total_weight => sum => :expec_per_pmt, nrow => :nhits)
npmts = get_pmt_count(target)


expec_per_pmt = rightjoin(expec_per_pmt, DataFrame(pmt_id=1:get_pmt_count(target), rel_angle=cos_rel_angles), on=:pmt_id)
expec_per_pmt[!, :weight_ratio] = expec_per_pmt[:, :nhits] ./ expec_per_pmt[:, :expec_per_pmt]
push!(dfs, expec_per_pmt)
end


all_dfs = vcat(dfs...)


@df all_dfs scatter(:rel_angle, :nhits)

plot(expec_per_pmt[:, :weight_ratio])

ids = Set(expec_per_pmt[:, :pmt_id])

for i in 1:16
    if !(i in ids)
        push!(expec_per_pmt, (pmt_id=>i, expec_per_pmt=>0))
    end
end


#
for distance in distances

    target = MultiPMTDetector(@SVector[distance, 0f0, 0f0], target_radius, pmt_area,
        make_pom_pmt_coordinates(Float32))

    prop_source = PointlikeCherenkovEmitter(particle, medium, Int64(1E12), (300f0, 800f0))

    res, nph_sim = propagate_photons(prop_source, target, medium)
    res = make_hits_from_photons(res, source, target, medium)
    groups = groupby(res, :pmt_id)

    expec_per_pmt = combine(groups, :total_weight => sum => :expec_per_pmt)


    #resampled_hits = combine(groups, [:time, :total_weight] => resample_simulation => :time)
resampled_hits

write_parquet("test_event.parquet", resampled_hits)


groupby(resampled_hits, :pmt_id)[2]


tres_tt = subtract_mean_tt(apply_tt(groups[5][:, :tres], STD_PMT_CONFIG.tt_dist), STD_PMT_CONFIG.tt_dist)
histogram(groups[5][:, :tres], weights=groups[5][:, :total_weight], xlim=(-10, 100), bins=-10:1:20)
histogram!(tres_tt, weights=groups[5][:, :total_weight], xlim=(-10, 100), bins=-10:1:20, alpha=0.8)
rs = resample_simulation(groups[5])



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
