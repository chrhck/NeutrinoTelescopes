using NeutrinoTelescopes
using Plots
using StaticArrays
using DataFrames
using StatsPlots
using Rotations
using Formatting
import Pipe: @pipe
using LinearAlgebra
using Distributions
using BenchmarkTools


function make_biolumi_sources(
    n_pos::Integer,
    n_ph::Integer,
    trange::Float64)
    sources = Vector{PointlikeTimeRangeEmitter}(undef, n_pos)

    for i in 1:n_pos

        pos_x::Float32 = 0
        pos_y::Float32 = 0
        pos_z::Float32 = 0

        if i < 20
            pos_z = rand([-1, 1]) * rand(Uniform(0.3, 3))
            pos_x = rand(Uniform(0.1, 1))
            pos_y = rand(Uniform(-0.2, 0.2))


        else
            pos_z = rand(Normal(0, 1))
            pos_x = rand(Uniform(0.5, 5))
            pos_y = rand(Uniform(-1, 1))
        end

        sources[i] = PointlikeTimeRangeEmitter(
            @SVector[pos_x, pos_y, pos_z],
            (0., trange),
            Int64(n_ph)
        )
    end

    return sources
end

function make_random_sources(
    n_pos::Integer,
    n_ph::Integer,
    trange::Real,
    radius::Real)
    sources = Vector{PointlikeTimeRangeEmitter}(undef, n_pos)


    radii = rand(Uniform(0, 1), n_pos) .^(1/3) .* (radius - 0.3) .+ 0.3
    thetas = acos.(rand(Uniform(-1, 1), n_pos))
    phis = rand(Uniform(0, 2*π), n_pos)

    pos = Float32.(radii) .* sph_to_cart.(Float32.(thetas), Float32.(phis))

    sources = PointlikeTimeRangeEmitter.(
        pos,
        Ref((0., trange)),
        Ref(Int64(n_ph))
    )

    return sources
end

function lc_trigger(sorted_hits, time_window)

    triggers = []

    i = 1
    while i < nrow(sorted_hits)

        lc_flag = false

        j = i+1
        while j <= nrow(sorted_hits)
            if (sorted_hits[j, :time] - sorted_hits[i, :time]) <= time_window
                lc_flag = true
            else
                break
            end
            j += 1
        end

        if !lc_flag
            i = j
            continue
        end

        if length(unique(sorted_hits[i:(j-1), :pmt_id])) >= 2
            push!(triggers, sorted_hits[i:(j-1), :])
        end

        i = j
    end


    return triggers
end

function plot_sources(sources)

    scatter([0], [0], [0], marksersize=10, markercolor=:black,
    xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5))

    scatter!(
        [src.position[1] for src in sources],
        [src.position[2] for src in sources],
        [src.position[3] for src in sources]
        )

    plot!([0, 0], [0, 0], [-5, 5])
end


function calc_trigger(hits, timewindow)
    sorted_hits = sort(hits, [:time])
    triggers = lc_trigger(sorted_hits, timewindow)
    coincs = []
    for trigger in triggers
        push!(coincs, unique(trigger[:, :pmt_id]))
    end
    return coincs
end


function sim_biolumi(target, sources)

    photons = propagate_photons(sources, target, medium, mono_spec)
    hits = make_hits_from_photons(photons, target, medium, orientation)
    all_hits = resample_simulation(hits)
    all_hits[!, :time] = convert.(Float64, all_hits[:, :time])
    return all_hits

end



function run_sim(target, trange::Number, n::Integer, bio_nph::Number, scale::Number)
    df = DataFrame(bio=Float64[], rnd=Float64[])
    coincs = []
    coincs_rnd = []
    for _ in 1:n

        sources = make_biolumi_sources(100, Int64(bio_nph), trange)
        rnd_sources = make_random_sources(200, Int64(ceil(bio_nph*scale)), trange, 5)

        all_hits = sim_biolumi(target, sources)
        rate = nrow(all_hits) / trange * 1E9
      

        all_hits_rnd = sim_biolumi(target, rnd_sources)
        rate_rnd = nrow(all_hits_rnd) / trange * 1E9

        if get_pmt_count(target) > 1
            triggers = calc_trigger(all_hits)
            triggers_rnd = calc_trigger(all_hits_rnd)

            push!(coincs, triggers)
            push!(coincs_rnd, triggers_rnd)
        end
        push!(df, (rate, rate_rnd))

    end

    return df, coincs, coincs_rnd
end



medium = make_cascadia_medium_properties(0.99f0)
pmt_area = Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = MultiPMTDetector(
    @SVector[0.0f0, 0f0,  0.f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32))


pmt_coord = @SMatrix zeros(Float32, 2, 1)

target_1pmt = MultiPMTDetector(
    @SVector[0.0f0, 0f0,  0.0f0],
    target_radius,
    pmt_area,
    pmt_coord
)

mono_spec = Monochromatic(420f0)
orientation = RotMatrix3(I)

trange = 1E8


rates, _, _ = run_sim(target_1pmt, trange, 100, 1E7, 1.985)
histogram(log10.(rates[:, :bio]),  alpha=0.7, )
histogram!(log10.(rates[:, :rnd]),  alpha=0.7,)


mean_rates = mapcols(mean, rates)
mean_rates[:, :bio] / mean_rates[:, :rnd]

mean_rates

rates, t, trnd = run_sim(target, trange, 100, 1E7, 1.985)

mean_rates_multi = mapcols(mean, rates)
mean_rates_multi[:, :bio] / mean_rates_multi[:, :rnd]


coinc_levels = vcat([length.(c) for c in t]...)
coinc_levels_rnd = vcat([length.(c) for c in trnd]...)

scatterhist(
    coinc_levels, weights=fill((1E9/(trange * length(t))),
    length(coinc_levels)),
    yscale=:log10,
    bins=1:7,
    ylim=(1E-2, 1E7),
    label="Biolumi",
    xlabel="Coincidence Level",
    title=format("Single-PMT Rate: {:.2f} kHz", mean_rates[1, :bio] / 1000))
p = scatterhist!(
    coinc_levels_rnd,
    weights=fill((1E9/(trange * length(trnd))),
    length(coinc_levels_rnd)),
    yscale=:log10,
    label="Random",
    bins=1:7,
    ylabel="Rate (Hz)")

savefig(p, joinpath(@__DIR__, "../figures/biolumi_coinc_exam.png"))


sorted_hits = sort(all_hits, [:time])

counts = combine(
    groupby(sorted_hits, :pmt_id),
    nrow => :counts
)

counts[!, :rates] = counts[:, :counts] / trange * 1E9
counts

triggers = lc_trigger(sorted_hits, 20)
coincs = []
for trigger in triggers
    push!(coincs, unique(trigger[:, :pmt_id]))
end
histogram(length.(coincs), yscale=:log10, weights=fill(1E9 / 1E9, length(coincs)))
