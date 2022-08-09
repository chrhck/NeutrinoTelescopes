module Types

using StaticArrays

export ParticleType, EPlus, EMinus, Gamma
export Particle

abstract type ParticleType end

struct EPlus <:ParticleType end
struct EMinus <:ParticleType end
struct Gamma <:ParticleType end

pdg_code(::Type{EPlus}) = -11
pdg_code(::Type{EMinus}) = 11
pdg_code(::Type{Gamma}) = 22

mutable struct Particle{T, PType <: ParticleType}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    energy::T
    type::PType
end
end