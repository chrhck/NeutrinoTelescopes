using NeutrinoTelescopes
using StaticArrays
using Random
using DataFrames
using ProgressLogging
using Formatting
using JSON
using Arrow
using Distributions
using CairoMakie
using Rotations
using LinearAlgebra
using HDF5
using TerminalLoggers
using Logging: global_logger
using Sobol
using ArgParse

Base.@kwdef struct PhotonTable{T}
    hits::T
    energy::Float64
    distance::Float64
    dir_theta::Float64
    dir_phi::Float64
    pos_theta::Float64
    pos_phi::Float64
end


function save_photon_table(fname::AbstractString, res::PhotonTable)

    if isfile(fname)
        fid = h5open(fname, "r+")
        ds_offset = length(fid["photon_tables"])+1
        g = fid["photon_tables"]
    else
        fid = h5open(fname, "w")
        ds_offset = 1
        g = create_group(fid, "photon_tables")
    end

    
    ds_name = format("dataset_{:d}", ds_offset)
    g[ds_name] = Matrix{Float64}(res.hits[:, [:time, :pmt_id, :total_weight]])

    for name in fieldnames(typeof(res))
        if name == :hits
            continue
        end
        HDF5.attributes(g[ds_name])[String(name)] = getfield(res, name)
    end

    close(fid)
end


s = ArgParseSettings()
@add_arg_table s begin
    "--n_sims"
        help = "Number of simulations"
        arg_type = Int
        required = true
    "--n_skip"
        help = "Skip in Sobol sequence"
        arg_type = Int
        required = false
        default = 0
end
parsed_args = parse_args(ARGS, s)


function run_sim(parsed_args)

    #=
    parsed_args = Dict("n_sims"=>1, "n_skip"=>0)
    =#
    medium = make_cascadia_medium_properties(0.99f0)
    pmt_area=Float32((75e-3 / 2)^2*π)
    target_radius = 0.21f0

    outdir = joinpath(@__DIR__, "../assets/")

    dfs = []
    sim_params = []

    spectrum = CherenkovSpectrum((300f0, 800f0), 30, medium)

    oversample = 1.

    n_sims = parsed_args["n_sims"]
    n_skip = parsed_args["n_skip"]

    sobol = skip(
        SobolSeq(
            [2, log10(10), -1, 0],
            [5, log10(100), 1, 2*π]),
        n_sims+n_skip)

    global_logger(TerminalLogger(right_justify=120))

    @progress "Photon sims" for i in 1:n_sims

        pars = next!(sobol)
        energy = 10^pars[1]
        distance = Float32(10^pars[2])
        dir_costheta = pars[3]
        dir_phi = pars[4]

        target = MultiPMTDetector(
            @SVector[0f0, 0f0, 0f0],
            target_radius,
            pmt_area,
            make_pom_pmt_coordinates(Float32)
        )

        direction::SVector{3, Float32} = sph_to_cart(acos(dir_costheta), dir_phi)

        ppos =  @SVector[0.0f0, 0f0, distance]
        particle = Particle(
            ppos,
            direction,
            0f0,
            Float32(energy),
            PEMinus
        )

        oversample = 1.
        photons = DataFrame()

        while true
        
            prop_source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0); oversample=oversample)
            if prop_source.photons > 1E13
                println("More than 1E13 photons, skipping")
                break
            end
            photons = propagate_photons(prop_source, target, medium, spectrum)

            println(format("Distance {:.1f} Photons: {:d} Hits: {:d}", distance, prop_source.photons, nrow(photons)))
            if nrow(photons) > 100
                break
            end
            oversample *= 10
        end

        @progress "Resampling" for j in 1:100
            #=
            PMT positions are defined in a standard upright coordinate system centeres at the module
            Sample a random rotation matrix and rotate the pmts on the module accordingly.
            =#
            orientation = rand(RotMatrix3)
            hits = make_hits_from_photons(photons, target, medium, orientation)

            hits[!, :total_weight] .*= oversample
            if nrow(hits) < 10
                continue
            end

            #=
            hits = resample_simulation(hits; per_pmt=true, downsample=1/oversample)

            if nrow(hits) == 0
                continue
            end
            =#

            #=
            Rotating the module (active rotation) is equivalent to rotating the coordinate system
            (passive rotation). Hence rotate the position and the direction of the light source with the
            inverse rotation matrix to obtain a description in which the module axis is again aligned with ez
            =#
            direction_rot = orientation' * direction
            position_rot = orientation' * ppos

            position_rot_normed = position_rot ./ norm(position_rot)
            dir_theta, dir_phi = cart_to_sph(direction_rot)
            pos_theta, pos_phi = cart_to_sph(position_rot_normed)

            # Sanity check:

            if !((dot(ppos / norm(ppos), direction) ≈ dot(position_rot_normed, direction_rot)))
                error("Relative angle not perseved")
            end


            save_photon_table(
                joinpath(outdir, "photon_table.hd5"),
                PhotonTable(
                    hits=hits,
                    energy=energy,
                    distance=Float64(distance),
                    dir_theta=dir_theta,
                    dir_phi=dir_phi,
                    pos_theta=pos_theta,
                    pos_phi=pos_phi)
                )
        end
    end
end

run_sim(parsed_args)
