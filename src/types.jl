module Types

using StaticArrays

export ParticleTypes, PDGCodes
export Particle

const ParticleTypes = Set([:EMinus, :EPlus, :Gamma])
const PDGCodes = Dict(:EMinus => 11, :Eplus => -11, :Gamma => 22)


mutable struct Particle{T}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    energy::T
    type::Symbol
end
end