using NeutrinoTelescopes
using StaticArrays
using DataFrames
using LinearAlgebra
using Rotations
using Parquet
using HDF5
using CairoMakie

medium = make_cascadia_medium_properties(0.99f0)

function positions_on_line(
    n_modules,
    dz;
    x=0.,
    y=0.,
    z_start=-500.,
    target_radius=0.21f0,
    pmt_area=Float32((75e-3 / 2)^2*π),
    id_offset=0)
    targets = [
        MultiPMTDetector(
        @SVector[Float32(x), Float32(y), Float32(z_start+(i-1)*dz)],
        target_radius,
        pmt_area, 
        make_pom_pmt_coordinates(Float32), UInt16(i + id_offset))
        for i in 1:n_modules
    ]

    return targets
end


side_len = 50
targets = [
    positions_on_line(20, 50; x=0, y=0, z_start=-500, id_offset=0)
    positions_on_line(20, 50; x=side_len, y=0, z_start=-500, id_offset=20)
    positions_on_line(20, 50; x=side_len/2, y=sqrt(side_len^2 - (side_len/2)^2), z_start=-500, id_offset=40)
    ]

id_target_map = Dict([targ.module_id => targ for targ in targets])


zenith_angle = 90f0
azimuth_angle = 0f0

particle = Particle(
    @SVector[10f0, 10f0, 0.0f0],
    sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
    0f0,
    Float32(1E5),
    PEMinus
)
cher_spectrum = CherenkovSpectrum((300f0, 800f0), 30, medium)
prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))


setup = PhotonPropSetup([prop_source_ext], targets, medium, cher_spectrum)
photons = propagate_photons(setup)

orientation = RotMatrix3(I)
hits = make_hits_from_photons(photons, setup, orientation)
calc_total_weight!(hits, setup)
resampled_hits = resample_simulation(hits)
calc_time_residual!(resampled_hits, setup)
res_grp_pmt = groupby(resampled_hits, [:module_id, :pmt_id]);


geo = DataFrame([(
    module_id=Int64(target.module_id),
    pmt_id=Int64(pmt_id),
    x=target.position[1],
    y=target.position[2],
    z=target.position[3],
    pmt_theta=coord[1],
    pmt_phi=coord[2])
    for target in targets
    for (pmt_id, coord) in enumerate(eachcol(target.pmt_coordinates))]
)

geo

outfile = joinpath(@__DIR__, "../assets/event_test_3str.hd5")

resampled_hits

h5open(outfile, "w") do hdl 

    g = create_group(hdl, "event_1")
    
    write(g, "photons", Matrix{Float64}(resampled_hits))
    attributes(g)["energy"] = particle.energy
    attributes(g)["position"] = particle.position
    attributes(g)["direction"] = particle.direction
    attributes(g)["time"] = particle.time

    g = create_group(hdl, "geometry")
    write(g, "geo", geo |> Matrix)

end


write_parquet(joinpath(outdir, "geometry_test_3str.parquet"), geo)
write_parquet(joinpath(outdir, "event_test_3str.parquet"), resampled_hits)
