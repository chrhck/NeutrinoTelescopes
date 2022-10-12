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


medium = make_cascadia_medium_properties(0.99f0)
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0

outdir = joinpath(@__DIR__, "../assets/")

dfs = []
sim_params = []

spectrum = CherenkovSpectrum((300f0, 800f0), 30, medium)

oversample = 1.

Base.@kwdef struct PhotonTable{T}
    hits::T
    energy::Float64
    distance::Float64
    dir_theta::Float64
    dir_phi::Float64
    pos_theta::Float64
    pos_phi::Float64
end


log_energy_dist = Uniform(2, 6)
log_distance_dist = Uniform(0, log10(300))


results = Vector{PhotonTable{DataFrame}}()

global_logger(TerminalLogger(right_justify=120))

@progress "Photon sims" for i in 1:50

    distance = Float32(10^rand(log_distance_dist))
    energy = 10^rand(log_energy_dist)
    target = MultiPMTDetector(
        @SVector[0f0, 0f0, 0f0],
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float32)
    )


    dir_costheta = rand(Uniform(-1, 1))
    dir_phi = rand(Uniform(0, 2*π))
    direction::SVector{3, Float32} = sph_to_cart(acos(dir_costheta), dir_phi)

    ppos =  @SVector[0.0f0, 0f0, distance]
    particle = Particle(
        ppos,
        direction,
        0f0,
        Float32(energy),
        PEMinus
    )

    prop_source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
    photons = propagate_photons(prop_source, target, medium, spectrum)

    #=
    Arrow.write(
        joinpath(outdir, format("photons_{:d}", i)),
        photons;
        metadata=["target" => json(target), "source" => json(prop_source)])
    =#

    @progress "Resampling" for j in 1:100
        #=
        PMT positions are defined in a standard upright coordinate system centeres at the module
        Sample a random rotation matrix and rotate the pmts on the module accordingly.
        =#
        orientation = rand(RotMatrix3)
        hits = make_hits_from_photons(photons, target, medium, orientation)

        if nrow(hits) == 0
            continue
        end

        hits = resample_simulation(hits)

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

        push!(
            results,
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


function save_photon_tables(fname, res::AbstractVector{<:PhotonTable})
    h5open(fname, "w") do fid
        g = create_group(fid, "photon_tables")

        for (i, tab) in enumerate(res)
            ds_name = format("dataset_{:d}", i)
            g[ds_name] = Matrix(tab.hits)

            for name in fieldnames(eltype(res))
                if name == :hits
                    continue
                end
                HDF5.attributes(g[ds_name])[String(name)] = getfield(tab, name)
            end

        end
    end
end

save_photon_tables(joinpath(outdir, "photon_table.hd5"), results)
