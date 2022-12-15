using NeutrinoTelescopes
using StaticArrays
using DataFrames
using CairoMakie
using Rotations
using LinearAlgebra
using ProgressLogging
using Parquet
using Unitful
using TerminalLoggers
using Logging

global_logger(TerminalLogger(right_justify=120))

medium = make_cascadia_medium_properties(0.99f0)
mono_spectrum = Monochromatic(450.0f0)

target = RectangularDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    0.001f0,
    0.003f0,
    UInt16(1),
)


laser_pos = @SVector[0.0f0, 0.05f0, 0.0f0]

all_photons = []
@progress for d in [20, 40, 60, 100, 150]

    beam_dir = @SVector[0.0f0, laser_pos[2] / (d - laser_pos[3]), 1.0f0]
    beam_dir = beam_dir ./ norm(beam_dir)
    beam_divergence = Float32(asin(2E-3 / 5))

    # Emmit from slightly outside the module
    prop_source_pencil_beam = PencilEmitter(
        laser_pos,
        beam_dir,
        beam_divergence,
        0.0f0,
        Int64(1E11)
    )

    @progress for i in 1:300
        setup = PhotonPropSetup(prop_source_pencil_beam, target, medium, mono_spectrum, Int64(i))
        photons = propagate_photons(setup)
        calc_total_weight!(photons, setup)
        photons[:, :d] .= d
        push!(all_photons, photons)
    end


end

photons = reduce(vcat, all_photons)

incident_angle = acos.(dot.(Ref(@SVector[0, 0, 1]), -photons[:, :direction]))

photons[:, :incident_angle] = incident_angle
mask = incident_angle .< deg2rad(0.4)
photons[:, :fov] = mask

write_parquet("lidar_photons.parquet", photons)

#=
photons_sav = DataFrame(read_parquet("data/lidar_photons.parquet"))
photons_sav


begin
    fig = Figure()
    ga = fig[1, 1] = GridLayout(3, 3)
    li = CartesianIndices((3, 3))
    bins = 0:30:1500
    for (i, (key, grp)) in enumerate(pairs(groupby(photons_sav, :d)))
        ax = Axis(ga[Tuple(li[i])...], xlabel="Time (ns)", ylabel="PDF", yscale=log10, limits=(-10, 1510, 1E-7, 1),
            title="Focus distance: $(key[1])m")


        hist!(ax, Float64.(grp[:, :time]), weights=Float64.(grp[:, :abs_weight]), bins=bins, normalization=:pdf, fillto=1E-7)
    end
    fig
end
=#
