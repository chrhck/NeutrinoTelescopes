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

target_radius = (13/2)u"inch" /  u"m"  |> NoUnits |> Float32

target = DetectionSphere(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    target_radius,
    1,
    #Float32(3E-6),
    Float32(3E-4),
    UInt16(1)
)

laser_pos = @SVector[0.0f0, 0.05f0, target_radius+0.05f0]
# Meet laser beam at 20m distance


all_photons = []
for d in 20f0:20f0:100f0

    beam_dir = @SVector[0f0, laser_pos[2] / (d - laser_pos[3]), 1f0]
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


    for i in 1:10
        setup = PhotonPropSetup(prop_source_pencil_beam, target, medium, mono_spectrum, Int64(i))
        photons = propagate_photons(setup)
        calc_total_weight!(photons, setup)
        photons[:, :d] .= d
        push!(all_photons, photons)
    end

    
end

photons = reduce(vcat, all_photons)



rel_pos = (SVector{3, Float64}.(photons[:, :position]) .- (Ref(SVector{3, Float64}(target.position)))) ./ Float64(target.radius)
opening_angle = asin(sqrt(target.pmt_area / π) / target.radius)
rel_pos ./= norm.(rel_pos)

mask = acos.(clamp.(dot.(rel_pos, Ref(@SVector[0, 0, 1])), -1, 1)) .< opening_angle

photons[:, :rel_pos] = rel_pos
photons[:, :hit] = mask

begin
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, limits=(0, 110, 1E-6, 1E-3))
    bins = 0:5:100
    for (key, grp) in pairs(groupby(photons, :d))

        m = grp[:, :hit]
        #m = trues(nrow(grp))

        hist!(ax, grp[m, :time], weights=grp[m, :total_weight], bins=bins, normalization=:density, fillto = 1E-6)
    end
    fig
end


    





hist(photons[mask, :time], weight=photons[mask, :total_weight], bins=bins)
photons_sav = photons[:, [:time, ]]

scatter(photons[:, :time], acos.((clamp.(dot.(rel_pos, Ref(@SVector[0, 0, 1])), -1, 1))),
        axis=(limits=(0, 1000, 0, 0.1), )
)



write_parquet("lidar_photons.parquet", photons_sav)

fig = scatter(photons[1:1000, :position])
scatter!(photons[mask, :position], color=:red)
fig

rel_pos[mask]

photons[mask, :]