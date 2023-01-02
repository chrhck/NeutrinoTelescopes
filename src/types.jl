module Types

using StaticArrays

export ParticleType, PEPlus, PEMinus, PGamma
export Particle

abstract type ParticleType end

struct PEPlus <:ParticleType end
struct PEMinus <:ParticleType end
struct PGamma <:ParticleType end

pdg_code(::Type{PEPlus}) = -11
pdg_code(::Type{PEMinus}) = 11
pdg_code(::Type{PGamma}) = 22

mutable struct Particle{PT, DT, TT, ET, PType <: ParticleType}
    position::SVector{3,PT}
    direction::SVector{3,DT}
    time::TT
    energy::ET
    type::Type{PType}
end



end