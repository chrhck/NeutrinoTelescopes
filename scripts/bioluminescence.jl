using NeutrinoTelescopes
using CairoMakie
using StaticArrays
using DataFrames
using Rotations
using Formatting
import Pipe: @pipe
using LinearAlgebra
using Distributions
using Random
using BenchmarkTools
using StatsBase
using Parquet
using JSON
using Base.Iterators


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

function calc_coincs_from_trigger(sorted_hits, timewindow)

    triggers = lc_trigger(sorted_hits, timewindow)
    coincs = Vector{Int64}()
    for trigger in triggers
        push!(coincs, length(unique(trigger[:, :pmt_id])))
    end
    return coincs
end


function count_coinc_in_tw(sorted_hits, time_window)

    t_start = sorted_hits[1, :time]

    window_cnt = Dict{Int64, Set{Int64}}()

    for i in 1:nrow(sorted_hits)
        window_id = div((sorted_hits[i, :time] - t_start), time_window)
        if !haskey(window_cnt, window_id)
            window_cnt[window_id] = Set([])
        end

        push!(window_cnt[window_id], sorted_hits[i, :pmt_id])
    end

    return length.(values(window_cnt))

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


function sim_biolumi(target, sources)

    medium = make_cascadia_medium_properties(0.99f0)
    mono_spec = Monochromatic(420f0)
    orientation = RotMatrix3(I)

    photons = propagate_photons(sources, target, medium, mono_spec)
    hits = make_hits_from_photons(photons, target, medium, orientation)
    all_hits = resample_simulation(hits)
    all_hits[!, :time] = convert.(Float64, all_hits[:, :time])
    return all_hits

end


function run_sim(
        target,
        sources,
        trange::Number,
       )

    all_hits = sim_biolumi(target, sources)

    downsampling = 10 .^(0:0.1:2)

    results = []

    for ds in downsampling

        if ds ≈ 1
            hits = all_hits
        else
            n_sel = Int64(ceil(nrow(all_hits) / ds))
            hits = all_hits[shuffle(1:nrow(all_hits))[1:n_sel], :]
        end

        rate = nrow(hits) / trange * 1E9 # Rate in Hz

        if get_pmt_count(target) > 1
            windows = [10, 15, 20, 15]
            sorted_hits = sort(hits, [:time])

            for window in windows
                coincs_trigger = calc_coincs_from_trigger(sorted_hits, window)
                coincs_fixed_w = count_coinc_in_tw(sorted_hits, window)

                ntup = (
                    ds_rate=ds,
                    hit_rate=rate,
                    time_window=window,
                    coincs_trigger=coincs_trigger,
                    coincs_fixed_w=coincs_fixed_w)
                push!(results, ntup)
            end
        else
            ntup = (
                ds_rate=ds,
                hit_rate=rate,
            )
            push!(results, ntup)

        end

    end

    return DataFrame(results)
end



pmt_area = Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = MultiPMTDetector(
    @SVector[0.0f0, 0f0,  0.f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32))

    make_pom_pmt_coordinates(Float32)

pmt_coord = @SMatrix zeros(Float32, 2, 1)

target_1pmt = MultiPMTDetector(
    @SVector[0.0f0, 0f0,  0.0f0],
    target_radius,
    pmt_area,
    pmt_coord
)

pmt_coord

trange = 1E7

function read_sources(path, trange, nph)
    bio_pos = DataFrame(read_parquet(path))
    bio_sources = Vector{PointlikeTimeRangeEmitter}()
    for i in 1:nrow(bio_pos)
        position = SVector{3, Float32}(Vector{Float32}(bio_pos[i, [:x, :y, :z]]))
        sources = PointlikeTimeRangeEmitter(position, (0., trange), Int64(ceil(nph)))
        push!(bio_sources, sources)
    end
    bio_sources
end


Random.seed!(31338)
bio_sources = [make_biolumi_sources(100, Int64(1E7), trange) for _ in 1:100]
write_parquet(joinpath(@__DIR__, "../assets/bio_sources.parquet"),
    DataFrame([(x=src.position[1], y=src.position[2], z=src.position[3]) for sources in bio_sources for src in sources]))


bio_sources = collect(
    partition(
        read_sources(joinpath(@__DIR__, "../assets/bio_sources.parquet"), trange, 1E7),
        100
    )
)

nph_rnd = Int64(ceil(4.035E7))
Random.seed!(31338)
rnd_sources = [make_random_sources(100, nph_rnd, trange, 5) for _ in 1:100]
write_parquet(joinpath(@__DIR__, "../assets/rnd_sources.parquet"),
    DataFrame([(x=src.position[1], y=src.position[2], z=src.position[3]) for sources in rnd_sources for src in sources]))

rnd_sources = collect(
    partition(
        read_sources(joinpath(@__DIR__, "../assets/rnd_sources.parquet"), trange, nph_rnd),
        100
    )
)


n_sim = 25

results_bio_1pmt = vcat(
    [run_sim(target_1pmt, sources, trange) for sources in bio_sources[1:n_sim]]...
    )
results_rnd_1pmt = vcat(
    [run_sim(target_1pmt, sources, trange) for sources in rnd_sources[1:n_sim]]...
    )

rates_bio = groupby(results_bio_1pmt, :ds_rate)[(1.0, )][:, :hit_rate]
rates_rnd = groupby(results_rnd_1pmt, :ds_rate)[(1.0, )][:, :hit_rate]

f = Figure()
ax = Axis(f[1, 1],
    title = "Single PMT-Rates",
    xlabel = "Log10(Rate)",
    ylabel = "Count"
)

hist!(ax, log10.(rates_bio), label="Bio")
hist!(ax, log10.(rates_rnd), label="Random")
axislegend(ax)
f

mean_hit_rate_1pmt_bio = combine(groupby(results_bio_1pmt, :ds_rate), :hit_rate => mean)
mean_hit_rate_1pmt_rnd = combine(groupby(results_rnd_1pmt, :ds_rate), :hit_rate => mean)

scale_factor = mean_hit_rate_1pmt_bio[1, :hit_rate_mean] / mean_hit_rate_1pmt_rnd[1, :hit_rate_mean]

n_sim = 5

results_bio = vcat(
    [run_sim(target, sources, trange) for sources in bio_sources[1:n_sim]]...
    )
results_rnd = vcat(
    [run_sim(target, sources, trange) for sources in rnd_sources[1:n_sim]]...
    )


function count_lc_levels(a)
    return [counts(vcat(a...), 2:10)]
end


function make_all_coinc_rate_plot(res...)

    f = Figure()
    ax = Axis(f[1, 1],
        title = "",
        xlabel = "Single PMT Rate",
        ylabel = "LC Rate",
        yscale = log10,
        xscale = log10,
        yminorticks = IntervalsBetween(8),
        yminorticksvisible = true,
        yminorgridvisible = true,


    )

    ylims!(ax, (1, 1E7))
    xlims!(ax, (1E4, 1E6))


    for (ls, (result_df, mean_hit_rate_1pmt)) in zip([:solid, :dashed], res)
        grpd_tw = groupby(groupby(result_df, :time_window)[3], :ds_rate)
        coinc_trigger = combine(grpd_tw, :coincs_trigger => count_lc_levels => AsTable)
        coinc_trigger = innerjoin(coinc_trigger, mean_hit_rate_1pmt, on=:ds_rate)


        for (i, lc_level) in enumerate(2:5)
            col_sym = Symbol(format("x{:d}", i))

            lines!(
                ax,
                coinc_trigger[:, :hit_rate_mean],
                coinc_trigger[:, col_sym] .* (1E9 / (trange * n_sim)),
                label=string(lc_level ),
                linestyle=ls
                )

        end
    end

    f[1, 2] = Legend(f, ax, "LC Level", framevisible = false)

    f
end



make_all_coinc_rate_plot(
    (results_bio, mean_hit_rate_1pmt_bio),
    (results_rnd, mean_hit_rate_1pmt_rnd))



f = Figure()

ax = Axis(f[1, 1])

make_coinc_rate_plot(results_rnd, mean_hit_rate_1pmt_rnd)


d = groupby(joined, :hit_rate)[15]
grpd = groupby(d, :time_window)

p = plot()
for key in keys(grpd)
    combined_coincs = vcat(grpd[key][:, :coincs_trigger]...)

    weights = fill(1E9 / trange, length(combined_coincs))

    scatterhist!(
        p,
        combined_coincs,
        weights=weights,
        label=string(key[1]),
        yscale=:log10,
        bins = 1:9,
        yrange=(1E-1, 1E8),
        yticks = 10 .^ (0:2:8)
        )
end

p
@df histogram(:coincs_trigger, groupby=:timewindow, yscale=:log10)



results_bio[1][1]


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
