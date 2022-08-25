module Detection
using StaticArrays
using CSV
using DataFrames
using Interpolations
using Unitful
using LinearAlgebra
using Base.Iterators

using ...Utils


export PhotonTarget, DetectionSphere, p_one_pmt_acc, MultiPMTDetector, make_pom_pmt_coordinates, get_pmt_count
export check_pmt_hit
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


struct MultiPMTDetector{T<:Real, N, L} <: PhotonTarget{T}
    position::SVector{3,T}
    radius::T
    pmt_area::T
    pmt_coordinates::SMatrix{2, N, T, L}
end

get_pmt_count(det::DetectionSphere) = 1
get_pmt_count(det::MultiPMTDetector) = size(det.pmt_coordinates, 2)

check_pmt_hit(::SVector{3, <:Real}, det::DetectionSphere) = 1

function check_pmt_hit(position::SVector{3, <:Real}, det::MultiPMTDetector{<:Real}) 
    rel_pos = (position .- det.position) ./ det.radius
    pmt_radius = sqrt(det.pmt_area / π) 
    opening_angle = asin(pmt_radius / det.radius)
    
    for (i, (det_θ, det_ϕ)) in enumerate(eachcol(det.pmt_coordinates))

        pmt_cart = sph_to_cart(det_θ, det_ϕ)
        rel_pos_rot = apply_rot(pmt_cart, SA[0., 0., 1.], rel_pos)

        if acos(rel_pos_rot[3]) < opening_angle
            return i
        end
    end
    return 0
end



function make_pom_pmt_coordinates(T::Type)

    coords = Matrix{T}(undef, 2, 16)

    # upper
    coords[1, 1:4] .= deg2rad(90 - 57.5)
    coords[2, 1:4] = (range(π/4; step=π/2, length=4))

    # upper 2
    coords[1, 5:8] .= deg2rad(90 - 25)
    coords[2, 5:8] = (range(0; step=π/2, length=4))

    # lower 2
    coords[1, 9:12] .= deg2rad(90 + 25)
    coords[2, 9:12] = (range(0; step=π/2, length=4))

    # lower
    coords[1, 13:16] .= deg2rad(90 + 57.5)
    coords[2, 13:16] = (range(π/4; step=π/2, length=4))

    R = calc_rot_matrix(SA[0., 0., 1.], SA[1., 0., 0.])
    @views for col in eachcol(coords)
        cart = sph_to_cart(col[1], col[2])
        col[:] .= cart_to_sph((R*cart)...)
   end

   return SMatrix{2, 16}(coords)
end




function area_acceptance(::SVector{3, <:Real}, target::DetectionSphere)
    total_pmt_area = target.n_pmts * target.pmt_area
    detector_surface = 4*π * target.radius^2

    return total_pmt_area / detector_surface
end

area_acceptance(::SVector{3, <:Real}, ::MultiPMTDetector) = 1

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