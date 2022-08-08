using Enzyme

using NeutrinoTelescopes.LightYield
using NeutrinoTelescopes.Types
using NeutrinoTelescopes.Utils
using NeutrinoTelescopes.Medium
using StaticArrays

function make_particle(theta::T, phi::T, x::T, y::T, z::T, log10E::T) where {T <: Real}

    dir = sph_to_cart(theta, phi)
    return Particle(SA[x, y, z], dir, 0., 10^(log10E), :EMinus)
end


medium = make_cascadia_medium_properties(Float64)

function calc_total_ly(particle)
    sources = particle_to_elongated_lightsource(particle, (0., 20.), 0.5, medium, (300., 800.))
    return sum([source.photons for source in sources])
end

part = make_particle(0., 0., 0., 0., 0., 4.)

calc_total_ly(part)

function f(theta::Real, phi::Real, x::Real, y::Real, z::Real, log10E::Real)
    part = make_particle(theta, phi, x, y, z, log10E)
    out::Float64 = calc_total_ly(part)
end


autodiff(Reverse, f, Active(0.), Active(0.), Active(0.), Active(0.), Active(0.), Active(5.))


autodiff(, Active(1.))


