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
using ArgParse

global_logger(TerminalLogger(right_justify=120))

function run_sim(; g, focus_distances, n_sims, output)


    medium = make_cascadia_medium_properties(Float32(g))
    mono_spectrum = Monochromatic(450.0f0)

    target = CircularDetector(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        ustrip(Float32, u"m", 1u"inch"),
        UInt16(1),
    )

    laser_pos = @SVector[0.0f0, 0.05f0, 0.0f0]

    all_photons = []
    @progress for d in focus_distances

        dy = (target.position[2] - laser_pos[2]) / (d - laser_pos[3])

        beam_dir = @SVector[0.0f0, dy, 1.0f0]
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

        @progress for i in 1:n_sims
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

    write_parquet(output, photons[:, [:time, :d, :abs_weight, :incident_angle]])

end

s = ArgParseSettings()
@add_arg_table s begin
    "--output"
    help = "Output filename"
    arg_type = String
    required = true
    "--n_sims"
    help = "Number of simulations"
    arg_type = Int
    required = true
    "--g"
    help = "Mean scattering angle"
    arg_type = Float32
    default = 0.99f0
    "--focus_distances"
    help = "Distances as which laser axis intersects PMT normal"
    arg_type = Float32
    nargs = '+'
end


parsed_args = parse_args(ARGS, s; as_symbols=true)
run_sim(; parsed_args...)




DataFrame(read_parquet(path))
#=
photons_sav = DataFrame(read_parquet("data/lidar_photons.parquet"))
photons_sav


begin
    fig = Figure()
    ga = fig[1, 1] = GridLayout(3, 3)
    li = CartesianIndices((3, 3))
    bins = 0:30:1500
    for (i, (key, grp)) in enumerate(pairs(groupby(photons, :d)))
        ax = Axis(ga[Tuple(li[i])...], xlabel="Time (ns)", ylabel="PDF", yscale=log10, limits=(-10, 1510, 1E-7, 1),
            title="Focus distance: $(key[1])m")

        m = grp[:, :fov]
        hist!(ax, Float64.(grp[m, :time]), weights=Float64.(grp[m, :abs_weight]), bins=bins, normalization=:pdf, fillto=1E-7)
    end
    fig
end
=#
