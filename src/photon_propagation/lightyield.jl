module LightYield

export LongitudinalParameters, LongitudinalParametersEMinus, LongitudinalParametersEPlus, LongitudinalParametersGamma
export LongitudinalParameterisation
export get_longitudinal_params
export MediumPropertiesWater, MediumPropertiesIce
export CherenkovTrackLengthParameters, CherenkovTrackLengthParametersEMinus, CherenkovTrackLengthParametersEPlus, CherenkovTrackLengthParametersGamma
export longitudinal_profile, cherenkov_track_length, cherenkov_counts, fractional_contrib_long
export particle_to_lightsource, particle_to_elongated_lightsource, particle_to_elongated_lightsource!
export total_lightyield

using Parameters: @with_kw
using SpecialFunctions: gamma
using StaticArrays
using QuadGK
using Sobol
using Zygote
using PhysicalConstants.CODATA2018
using Unitful
using PoissonRandom
using ..Spectral
using ..Medium
using ...Utils
using ...Types
using ..Emission

c_vac_m_p_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)

@with_kw struct LongitudinalParameters
    alpha::Float64
    beta::Float64
    b::Float64
end

const LongitudinalParametersEMinus = LongitudinalParameters(alpha=2.01849, beta=1.45469, b=0.63207)
const LongitudinalParametersEPlus = LongitudinalParameters(alpha=2.00035, beta=1.45501, b=0.63008)
const LongitudinalParametersGamma = LongitudinalParameters(alpha=2.83923, beta=1.34031, b=0.64526)

@with_kw struct LongitudinalParameterisation{T <: Real}
    a::T
    b::T
    lrad::T
end

function LongitudinalParameterisation(energy::T, medium::MediumProperties, long_pars::LongitudinalParameters) where {T <:Real}
    b = T(long_parameter_b_edep(energy, long_pars))
    a = T(long_parameter_a_edep(energy, long_pars))

    unit_conv = 10 # g/cm^2 / "kg/m^3" in m    
    lrad = radiation_length(medium) / density(medium) * unit_conv

    LongitudinalParameterisation(a, b, lrad)
end

LongitudinalParameterisation(energy::T, medium::MediumProperties, ::Type{ptype}) where {T <:Real, ptype <: ParticleType} = LongitudinalParameterisation(
    energy, medium, get_longitudinal_params(ptype))



@with_kw struct CherenkovTrackLengthParameters
    alpha::Float64 # cm
    beta::Float64
    alpha_dev::Float64 # cm
    beta_dev::Float64
end

const CherenkovTrackLengthParametersEMinus = CherenkovTrackLengthParameters(
    alpha=5.3207078881,
    beta=1.00000211,
    alpha_dev=0.0578170887,
    beta_dev=0.5
)

const CherenkovTrackLengthParametersEPlus = CherenkovTrackLengthParameters(
    alpha=5.3211320598,
    beta=0.99999254,
    alpha_dev=0.0573419669,
    beta_dev=0.5
)

const CherenkovTrackLengthParametersGamma = CherenkovTrackLengthParameters(
    alpha=5.3208540905,
    beta=0.99999877,
    alpha_dev=0.0578170887,
    beta_dev=5.66586567
)

@with_kw struct LightyieldParametrisation
    longitudinal::LongitudinalParameters
    track_length::CherenkovTrackLengthParameters
end


get_longitudinal_params(::Type{PEPlus}) = LongitudinalParametersEPlus
get_longitudinal_params(::Type{PEMinus}) = LongitudinalParametersEMinus
get_longitudinal_params(::Type{PGamma}) = LongitudinalParametersGamma

get_track_length_params(::Type{PEPlus}) = CherenkovTrackLengthParametersEPlus
get_track_length_params(::Type{PEMinus}) = CherenkovTrackLengthParametersEMinus
get_track_length_params(::Type{PGamma}) = CherenkovTrackLengthParametersGamma

function long_parameter_a_edep(
    energy::Real,
    long_pars::LongitudinalParameters
)
    long_pars.alpha + long_pars.beta * log10(energy)
end
long_parameter_a_edep(energy::Real, ::Type{ptype}) where {ptype <: ParticleType} = long_parameter_a_edep(energy, get_longitudinal_params(ptype))

long_parameter_b_edep(::Real, long_pars::LongitudinalParameters) = long_pars.b
long_parameter_b_edep(energy::Real, ::Type{ptype}) where {ptype <: ParticleType} = long_parameter_b_edep(energy, get_longitudinal_params(ptype))


"""
    longitudinal_profile(z::Real, medium::MediumProperties, long_pars::LongitudinalParameters)
    
Longitudinal shower profile (PDF) at shower depth z ( in m )
"""
function longitudinal_profile(z::Real, long_param::LongitudinalParameterisation)

    a = long_param.a
    b = long_param.b
    lrad = long_param.lrad

    t = z / lrad

    b * ((t * b)^(a - 1.0) * exp(-(t * b)) / gamma(a))
end

"""
    longitudinal_profile(energy::Real, z::Real, medium::MediumProperties, long_pars::LongitudinalParameters)
    
energy in GeV, z in m,
"""

function longitudinal_profile(
    energy::Real, z::Real, medium::MediumProperties, long_pars::LongitudinalParameters)
    long_param = LongitudinalParameterisation(energy, medium, long_pars)
    longitudinal_profile(z, long_param)

end

function longitudinal_profile(
    energy, z, medium, ::Type{ptype}) where {ptype <: ParticleType}
    longitudinal_profile(energy, z, medium, get_longitudinal_params(ptype))
end

"""
    gamma_cdf(a, b, z)

Cumulative Gamma distribution
\$ int_0^z Gamma(a, b) \$
"""
gamma_cdf(a, b, z) = 1. - gamma(a, b*z) / gamma(a)


"""
    integral_long_profile(z_low::Real, z_high::Real, long_param::LongitudinalParameterisation)

Integral of the longitudinal profile from shower depth z_low (in m) to z_high
"""
function integral_long_profile(z_low::Real, z_high::Real, long_param::LongitudinalParameterisation)
    a = long_param.a
    b = long_param.b
    lrad = long_param.lrad

    t_low = z_low / lrad
    t_high = z_high / lrad
    gamma_cdf(a, b, t_high) - gamma_cdf(a, b, t_low)
end

function integral_long_profile(energy::Real, z_low::Real, z_high::Real, medium::MediumProperties, long_pars::LongitudinalParameters)
    long_param = LongitudinalParameterisation(energy, medium, long_pars)
    integral_long_profile(z_low, z_high, long_param)    
end

function integral_long_profile(energy::Real, z_low::Real, z_high::Real, medium::MediumProperties, ::Type{ptype}) where {ptype <: ParticleType}
    integral_long_profile(energy, z_low, z_high, medium, get_longitudinal_params(ptype))
end


function fractional_contrib_long!(
    z_grid::AbstractVector{T},
    long_param::LongitudinalParameterisation,
    output::Union{Zygote.Buffer,AbstractVector{T}}
) where {T<:Real}
    if length(z_grid) != length(output)
        error("Grid and output are not of the same length")
    end

    norm = integral_long_profile(z_grid[1], z_grid[end], long_param)

    output[1] = 0
    @inbounds for i in 1:size(z_grid, 1)-1
        output[i+1] = (
            1 / norm * integral_long_profile(z_grid[i], z_grid[i+1], long_param)
        )
    end
    output
end

function fractional_contrib_long!(
    energy::Real,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    long_pars::LongitudinalParameters,
    output::Union{Zygote.Buffer,AbstractVector{T}}
) where {T<:Real}

    long_param = LongitudinalParameterisation(energy, medium, long_pars)
    fractional_contrib_long!(z_grid, long_param, output)
end

function fractional_contrib_long!(
    energy,
    z_grid,
    medium,
    ::Type{ptype},
    output) where {ptype <: ParticleType}
    fractional_contrib_long!(energy, z_grid, medium, get_longitudinal_params(ptype), output)
end

function fractional_contrib_long(
    energy::Real,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    pars_or_ptype::Union{LongitudinalParameters, ptype}
) where {T<:Real, ptype}
    output = similar(z_grid)
    fractional_contrib_long!(energy, z_grid, medium, pars_or_ptype, output)
end



function cherenkov_track_length_dev(energy::Real, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha_dev * energy^track_len_params.beta_dev
end
cherenkov_track_length_dev(energy::Real, ::Type{ptype}) where {ptype <: ParticleType} = cherenkov_track_length_dev(energy, get_track_length_params(ptype))

"""
    function cherenkov_track_length(energy::Real, track_len_params::CherenkovTrackLengthParameters)

energy in GeV

returns track length in m
"""

function cherenkov_track_length(energy::Real, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha * energy^track_len_params.beta
end
cherenkov_track_length(energy::Real, ::Type{ptype}) where {ptype <: ParticleType} = cherenkov_track_length(energy, get_track_length_params(ptype))


function total_lightyield(
    particle::Particle{T},
    medium::MediumProperties,
    wl_range::Tuple{T,T}
) where {T <:Real}
    total_contrib = (
        frank_tamm_norm(wl_range, wl -> get_refractive_index(wl, medium)) *
        cherenkov_track_length.(particle.energy, particle.type)
    )
    total_contrib
end



function particle_to_lightsource(
    particle::Particle{T},
    medium::MediumProperties,
    wl_range::Tuple{T,T}
) where {T<:Real}

    total_contrib = total_lightyield(particle, medium, wl_range)

    CherenkovSegment(
        particle.position,
        particle.direction,
        particle.time,
        total_contrib)


end

function particle_to_elongated_lightsource!(
    particle::Particle{T},
    int_grid::AbstractArray{T},
    medium::MediumProperties,
    wl_range::Tuple{T,T},
    output::Union{Zygote.Buffer,AbstractVector{PointlikeCherenkovEmitter{T, 20}}},
) where {T<:Real}


    """
    s = SobolSeq([range_cm[1]], [range_cm[2]])

    n_steps = Int64(ceil(ustrip(Unitful.NoUnits, (range[2] - range[1]) / precision)))
    int_grid = sort!(vec(reduce(hcat, next!(s) for i in 1:n_steps)))u"cm"
    """

    spectrum = CherenkovSpectrum(wl_range, 20, medium)

    n_steps = length(int_grid)

    fractional_contrib_vec = Vector{T}(undef, n_steps)

    if typeof(output) <: Zygote.Buffer
        fractional_contrib = Zygote.Buffer(fractional_contrib_vec)
    else
        fractional_contrib = fractional_contrib_vec
    end

    fractional_contrib_long!(particle.energy, int_grid, medium, particle.type, fractional_contrib)

    total_contrib = (
        frank_tamm_norm(wl_range, wl -> get_refractive_index(wl, medium)) *
        cherenkov_track_length(particle.energy, particle.type)
    )


    step_along = [T(0.5) * (int_grid[i] + int_grid[i+1]) for i in 1:(n_steps-1)]

    for i in 2:n_steps
        this_pos = particle.position .+ step_along[i-1] .* particle.direction
        this_time = particle.time + step_along[i-1] / T(c_vac_m_p_ns)

        this_nph = pois_rand(total_contrib * fractional_contrib[i])

        this_src = PointlikeCherenkovEmitter(
            this_pos,
            particle.direction,
            this_time,
            this_nph,
            spectrum)

        output[i-1] = this_src
    end

    output
end

function particle_to_elongated_lightsource(
    particle::Particle{T},
    len_range::Tuple{T,T},
    precision::T,
    medium::MediumProperties,
    wl_range::Tuple{T,T}
) where {T<:Real}

    int_grid = range(len_range[1], len_range[2], step=precision)
    n_steps = size(int_grid, 1)
    output = Vector{PointlikeCherenkovEmitter{T, 20}}(undef, n_steps - 1)
    particle_to_elongated_lightsource!(particle, int_grid, medium, wl_range, output)
end






end