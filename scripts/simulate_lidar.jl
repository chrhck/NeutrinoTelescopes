using NeutrinoTelescopes
using StaticArrays
using DataFrames
using CairoMakie
using Rotations
using LinearAlgebra
using ProgressLogging
using Parquet
using Unitful

medium = make_cascadia_medium_properties(0.99f0)
mono_spectrum = Monochromatic(450.0f0)

target = RectangularDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    0.001f0,
    0.003f0,
    UInt16(1),
)


laser_pos = @SVector[0.0f0, 0.05f0, 0f0]
# Meet laser beam at 20m distance

# TODO ISEC ASSUMES normal IS e_z
function intersect_rect(photons, normal, center, dx, dy)
    dirnormal = dot.(photons[:, :direction], Ref(normal))
    d = dot.(Ref(center) .- photons[:, :position], Ref(normal)) ./ dirnormal

    isec_point = reduce(hcat, photons[:, :position] .+ photons[:, :direction] .* d)
    isec = (abs.(isec_point[1, :] .- center[1]) .< dx) .& (abs.((isec_point[2, :] .- center[2])) .< dy)
    isec .&= (dirnormal .< 0)

    return isec
end



all_photons = []
@progress for d in [20, 40, 60, 100, 150]

    beam_dir = @SVector[0f0, laser_pos[2] / (d - laser_pos[3]), 1f0]
    beam_dir = beam_dir ./ norm(beam_dir)
    beam_divergence = Float32(asin(2E-3 / 5))

    # Emmit from slightly outside the module
    prop_source_pencil_beam = PencilEmitter(
        laser_pos,
        beam_dir,
        beam_divergence,
        0.0f0,
        Int64(1E12)
    )

    @progress for i in 1:2
        setup = PhotonPropSetup(prop_source_pencil_beam, target, medium, mono_spectrum, Int64(i))
        photons = propagate_photons(setup)
        calc_total_weight!(photons, setup)
        photons[:, :d] .= d
        push!(all_photons, photons)
    end

    
end

photons = reduce(vcat, all_photons)

incident_angle = acos.(dot.(Ref(@SVector[0, 0, 1]), -photons[:, :direction]))
mask = incident_angle .< deg2rad(0.4)
photons[:, :fov] = mask

photons_sav = photons[photons[:, :fov], [:time, :d, :abs_weight]]
write_parquet("lidar_photons.parquet", photons_sav)


photons[photons[:, :fov_and_hit], :position]


begin
    fig = Figure()
    ga = fig[1, 1] = GridLayout(3, 3)
    li = CartesianIndices((3, 3))
    bins = 0:50:1500
    for (i, (key, grp)) in enumerate(pairs(groupby(photons, :d)))
        ax = Axis(ga[Tuple(li[i])...], xlabel="Time (ns)", ylabel="PDF", yscale=log10, limits=(-10, 1510, 1E-7, 1),
                 title="Focus distance: $(key[1])m")
        m = grp[:, :fov_and_hit]
        #m = trues(nrow(grp))

        hist!(ax, grp[m, :time], weights=grp[m, :total_weight], bins=bins, normalization=:pdf, fillto = 1E-7)
    end
    fig
end


photons_sav = photons[photons[:, :fov_and_hit], [:time, :d, :abs_weight]]

write_parquet("lidar_photons.parquet", photons_sav)

fig = scatter(photons[1:1000, :position])
scatter!(photons[mask, :position], color=:red)
fig

rel_pos[mask]

photons[mask, :]

absorption_length(450f0, medium)
scattering_length(450f0, medium)

0.3 / refractive_index(450f0, medium) * 500
