module Detection
using StaticArrays
using CSV
using DataFrames
using Interpolations
using Unitful
using Base.Iterators


export PhotonTarget, DetectionSphere, p_one_pmt_acc
export make_detector_cube, make_targets, make_detector_hex
export area_acceptance

const PROJECT_ROOT = pkgdir(Detection)

abstract type PhotonTarget{T<:Real} end

struct DetectionSphere{T<:Real} <: PhotonTarget{T}
    position::SVector{3,T}
    radius::T
    n_pmts::Int64
    pmt_area::T
end

function area_acceptance(target::DetectionSphere)
    total_pmt_area = target.n_pmts * target.pmt_area
    detector_surface = 4*π * target.radius^2

    return total_pmt_area / detector_surface
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



function make_detector_cube(nx, ny, nz, spacing_vert::T, spacing_hor::T) where {T <: Real}

    positions = Vector{SVector{3,T}}(undef, nx * ny * nz)

    lin_ix = LinearIndices((1:nx, 1:ny, 1:nz))
    for (i, j, k) in product(1:nx, 1:ny, 1:nz)
        ix = lin_ix[i, j, k]
        positions[ix] = @SVector [-0.5 * spacing_hor * nx + (i - 1) * spacing_hor, -0.5 * spacing_hor * ny + (j - 1) * spacing_hor, -0.5 * spacing_vert * nz + (k - 1) * spacing_vert]
    end

    positions
end


function make_detector_hex(n_side::Integer, n_z::Integer, spacing_hor::Real, spacing_vert::Real, truncate::Integer=0)
    positions = []

    z_positions = range(0, n_z*spacing_vert; step=spacing_vert)

    for irow in 0:(n_side - truncate)
        i_this_row = 2 * (n_side - 1) - irow
        x_pos = range(-(i_this_row - 1) / 2 * spacing_hor, (i_this_row - 1) / 2 * spacing_hor; length=i_this_row)
        y = irow * spacing_hor * sqrt(3) / 2
        for (x, z) in product(x_pos, z_positions)
            push!(positions, SA[x, y, z])
        end
        if irow != 0
            x_pos = range(-(i_this_row - 1) / 2 * spacing_hor, (i_this_row - 1) / 2 * spacing_hor; length=i_this_row)
            y = -irow * spacing_hor * sqrt(3) / 2

            for (x, z) in product(x_pos, z_positions)
                push!(positions, SA[x, y, z])
            end
        end
    end

    positions
end



function make_targets(positions, n_pmts, pmt_area)
    map(pos -> DetectionSphere(pos, oftype(pmt_area, 0.21), n_pmts, pmt_area), positions)
end



end