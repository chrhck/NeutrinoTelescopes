module Detection
using StaticArrays
using CSV
using DataFrames
using Interpolations
using Unitful
using Base.Iterators


export PhotonTarget, DetectionSphere, p_one_pmt_acc
export make_detector_cube, make_targets

const PROJECT_ROOT = pkgdir(Detection)

abstract type PhotonTarget{T<:Real} end

struct DetectionSphere{T<:Real} <: PhotonTarget{T}
    position::SVector{3,T}
    radius::T
end


struct PMTWavelengthAcceptance
    interpolation::Interpolations.Extrapolation

    PMTWavelengthAcceptance(interpolation::Interpolations.Extrapolation) = error("default constructor disabled")
    function PMTWavelengthAcceptance(xs::AbstractVector, ys::AbstractVector)
        new(LinearInterpolation(xs, ys))
    end
end

(f::PMTWavelengthAcceptance)(wavelength::Real) = f.interpolation(wavelength)
(f::PMTWavelengthAcceptance)(wavelength::Unitful.Length) = f.interpolation(ustrip(u"nm", wavelength))


df = CSV.read(joinpath(PROJECT_ROOT, "assets/PMTAcc.csv",), DataFrame, header=["wavelength", "acceptance"])

p_one_pmt_acc = PMTWavelengthAcceptance(df[:, :wavelength], df[:, :acceptance])



function make_detector_cube(nx, ny, nz, spacing_vert, spacing_hor)

    positions = Vector{SVector{3,Float64}}(undef, nx * ny * nz)

    lin_ix = LinearIndices((1:nx, 1:ny, 1:nz))
    for (i, j, k) in product(1:nx, 1:ny, 1:nz)
        ix = lin_ix[i, j, k]
        positions[ix] = @SVector [-0.5 * spacing_hor * nx + (i - 1) * spacing_hor, -0.5 * spacing_hor * ny + (j - 1) * spacing_hor, -0.5 * spacing_vert * nz + (k - 1) * spacing_vert]
    end

    positions

end

function make_targets(positions)
    map(pos -> DetectionSphere(pos, 0.21), positions)
end




end