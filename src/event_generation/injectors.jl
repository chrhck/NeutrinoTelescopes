module Injectors

using Random
using StaticArrays
using Distributions
import Base.rand

using ...Types
using ...Utils


export sample_volume, inject
export Cylinder, Cuboid, VolumeType
export VolumeInjector, Injector
export ParticleTypeDistribution
export AngularDistribution, UniformAngularDistribution

"""
    VolumeType

Abstract type for volumes
"""
abstract type VolumeType end

"""
    Cylinder{T} <: VolumeType

Type for cylindrical volumes.
"""
struct Cylinder{T} <: VolumeType
    center::SVector{3,T}
    height::T
    radius::T
end

"""
    Cylinder{T} <: VolumeType

Type for cuboid volumes.
"""
struct Cuboid{T} <: VolumeType
    center::SVector{3,T}
    l_x::T
    l_y::T
    l_z::T
end

struct FixedPosition <: VolumeType
    position::SVector{3,T}

"""
    rand(::VolumeType)

Sample a random point in volume
"""
rand(::VolumeType) = error("Not implemented")
rand(vol::FixedPosition) = vol.position

function rand(vol::Cylinder{T}) where {T}
    uni = Uniform(-vol.height / 2, vol.height / 2)
    rng_z = rand(uni)

    rng_r = sqrt(rand(T) * vol.radius)
    rng_phi = rand(T) * 2 * π
    rng_x = rng_r * cos(rng_phi)
    rng_y = rng_r * sin(rng_phi)

    return SA{T}[rng_x, rng_y, rng_z] + vol.center

end

function rand(vol::Cuboid{T}) where {T}
    uni_x = Uniform(-vol.lx / 2, vol.lx / 2)
    uni_y = Uniform(-vol.ly / 2, vol.ly / 2)
    uni_z = Uniform(-vol.lz / 2, vol.lz / 2)
    return SA{T}[rand(uni_x), rand(uni_y), rand(uni_z)] + vol.center

end

abstract type AngularDistribution end

struct UniformAngularDistribution <: AngularDistribution end

function Base.rand(::UniformAngularDistribution)
    phi = rand() * 2 * π
    theta = acos(2 * rand() - 1)

    return sph_to_cart(theta, phi)
end


abstract type Injector end
struct VolumeInjector{T<:VolumeType,U<:UnivariateDistribution,W<:AngularDistribution} <: Injector
    volume::T
    e_dist::U
    type_dist::CategoricalSetDistribution{Symbol}
    angular_dist::W
end




function Base.rand(inj::VolumeInjector)
    pos = rand(inj.volume)
    energy = rand(inj.e_dist)
    ptype = rand(inj.type_dist)
    dir = rand(inj.angular_dist)

    return Particle(pos, dir, zero(eltype(pos)), energy, ptype)

end

end
