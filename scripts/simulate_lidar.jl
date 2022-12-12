using NeutrinoTelescopes
using StaticArrays
using DataFrames
using CairoMakie
using Rotations
using LinearAlgebra


medium = make_cascadia_medium_properties(0.99f0)
mono_spectrum = Monochromatic(450.0f0)


prop_source_pencil_beam = PencilEmitter(
    @SVector[0.0f0, 0.05f0, 0.1f0],
    @SVector[0.0f0, 0.0f0, 1.0f0],
    0.0f0,
    Int64(1E11)
)

target = DetectionSphere(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    0.05f0,
    1,
    Float32(3E-6),
    UInt16(1)
)


setup = PhotonPropSetup(prop_source_pencil_beam, target, medium, mono_spectrum)

photons = propagate_photons(setup)
calc_total_weight!(photons, setup)

rot = RotMatrix3(I)
make_hits_from_photons(photons, setup, rot)
photons[:, :position]

rel_pos = (Ref(target.position) .- photons[:, :position]) ./ target.radius

opening_angle = deg2rad(5)
any(acos.(clamp.(dot.(rel_pos, Ref(@SVector[0, 0, 1])), -1, 1)) .< opening_angle)

acos.(clamp.(dot.(rel_pos, Ref(@SVector[0, 0, 1])), -1, 1))

rel_pos

rel_pos_cart = cart_to_sph.(rel_pos) <= opening_angle




hist(photons[:, :time], weight=photons[:, :total_weight])

photons_sav = photons[:, :time]
