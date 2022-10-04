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

using Waterlily

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

divrem(1E10, 9)

prop_source_isospan = PointlikeTimeRangeEmitter(
    @SVector[0.0f0, 0f0, 3f0],
    (0f0, Float32(1E6)),
    Int64(1E9)
)

sources = split_source(prop_source_isospan, 3)


mono_spec = Monochromatic(420f0)
orientation = RotMatrix3(I)


function make_biolumi_sources(
    n_pos::Integer,
    n_ph::Integer)
    sources = Vector{PointlikeTimeRangeEmitter}(undef, n_pos)

    for i in 1:n_pos

        pos_x::Float32 = 0
        pos_y::Float32 = 0
        pos_z::Float32 = 0

        if i < 20
            pos_z = rand([-1, 1]) * rand(Uniform(0.3, 3))
            pos_x = rand(Uniform(0.3, 2))
            pos_y = rand(Uniform(-0.2, 0.2))


        else
            pos_z = rand(Normal(0, 1))
            pos_x = rand(Uniform(0.5, 5))
            pos_y = rand(Uniform(-1, 1))
        end

        sources[i] = PointlikeTimeRangeEmitter(
            @SVector[pos_x, pos_y, pos_z],
            (0f0, Float32(1E7)),
            Int64(n_ph)
        )
    end

    return sources
end


function sim_biolumi(target, sources)

    photons = propagate_photons(sources, target, medium, mono_spec)
    hits = make_hits_from_photons(photons, target, medium, orientation)
    all_hits = resample_simulation(hits)
    all_hits[!, :time] = convert.(Float64, all_hits[:, :time])
    return all_hits

end

sources = make_biolumi_sources()

scatter([0], [0], [0], marksersize=10, markercolor=:black,
xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5))

scatter!(
    [src.position[1] for src in sources],
    [src.position[2] for src in sources],
    [src.position[3] for src in sources]
    )

plot!([0, 0], [0, 0], [-5, 5])


all_hits = sim_biolumi(target_1pmt)

rate_1pmt = nrow(all_hits) / 1E7 * 1E9

all_hits = sim_biolumi(target)
sorted_hits = sort(all_hits, [:time])

counts = combine(
    groupby(sorted_hits, :pmt_id),
    nrow => :counts
)

counts[!, :rates] = counts[:, :counts] / 1E7 * 1E9
counts


triggers = lc_trigger(sorted_hits, 20)
coincs = []
for trigger in triggers
    push!(coincs, unique(trigger[:, :pmt_id]))
end
histogram(length.(coincs), yscale=:log10, weights=fill(1E9 / 1E7, length(coincs)))
