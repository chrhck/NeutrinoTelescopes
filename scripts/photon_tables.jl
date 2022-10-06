using NeutrinoTelescopes
using StaticArrays
using Random
using DataFrames
using ProgressLogging
using Formatting
using JSON
using Arrow

medium = make_cascadia_medium_properties(Float32)
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0

outdir = joinpath(@__DIR__, "../assets/")

dfs = []
sim_params = []


target = MultiPMTDetector(
    @SVector[distance, 0f0, 0f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32)
    )

distance = 10f0
energy = 1E5
@progress "Photon sims" for i in 1:10
    dir_costheta = rand(Uniform(-1, 1))
    dir_phi = rand(Uniform(0, 2*π))
    direction::SVector{3, Float32} = sph_to_cart(acos(dir_costheta), dir_phi)

    particle = Particle(
            @SVector[0.0f0, 0f0, 0.0f0],
            direction,
            0f0,
            Float32(energy),
            PEMinus
    )

    prop_source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
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
