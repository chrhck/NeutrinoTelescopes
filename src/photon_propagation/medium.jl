module Medium
using Unitful
using Base: @kwdef
using PhysicalConstants.CODATA2018
using Parquet
using DataFrames

using ...Utils

export make_cascadia_medium_properties
export salinity, pressure, temperature, vol_conc_small_part, vol_conc_large_part, radiation_length, material_density
export refractive_index, scattering_length, absorption_length, dispersion, group_velocity, cherenkov_angle
export mean_scattering_angle
export MediumProperties, WaterProperties

@unit ppm "ppm" Partspermillion 1 // 1000000 false
Unitful.register(Medium)

const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)

"""
    DIPPR105Params

Parameters for the DIPPR105 formula
"""
@kwdef struct DIPPR105Params
    A::Float64
    B::Float64
    C::Float64
    D::Float64
end

# DIPR105 Parameters from DDB
const DDBDIPR105Params = DIPPR105Params(A=0.14395, B=0.0112, C=649.727, D=0.05107)

"""
    DIPPR105(temperature::Real, params::DIPPR105Params=DDBDIPR105Params)

Use DPPIR105 formula to calculate water density as function of temperature.
temperature in K.

Reference: http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe?component=Water

Returns density in kg/m^3
"""
function DIPPR105(temperature::Real, params::DIPPR105Params=DDBDIPR105Params)
    return params.A / (params.B^(1 + (1 - temperature / params.C)^params.D))
end



abstract type MediumProperties{T<:Real} end


"""
    WaterProperties{T<:Real} <: MediumProperties{T}

Properties for a water-like medium. Use unitful constructor to create a value of this type.

### Fields:
-salinity -- Salinity (permille)
-pressure -- Pressure (atm)
-temperature -- Temperature (°C)
-vol_conc_small_part -- Volumetric concentrations of small particles (ppm)
-vol_conc_large_part -- Volumetric concentrations of large particles (ppm)
-radiation_length -- Radiation length (g/cm^2)
-density -- Density (kg/m^3)
-mean_scattering_angle -- Cosine of the mean scattering angle
"""
struct WaterProperties{T<:Real} <: MediumProperties{T}
    salinity::T # permille
    pressure::T # atm
    temperature::T #°C
    vol_conc_small_part::T # ppm
    vol_conc_large_part::T # ppm
    radiation_length::T # g / cm^2
    density::T # kg/m^3
    mean_scattering_angle::T
    quan_fry_params::Tuple{T, T, T, T}

    WaterProperties(::T, ::T, ::T, ::T, ::T, ::T, ::T, ::T, ::Tuple{T, T, T, T}) where {T} = error("Use unitful constructor")

    @doc """
            function WaterProperties(
                salinity::Unitful.Quantity{T},
                pressure::Unitful.Quantity{T},
                temperature::Unitful.Quantity{T},
                vol_conc_small_part::Unitful.Quantity{T},
                vol_conc_large_part::Unitful.Quantity{T},
                radiation_length::Unitful.Quantity{T}
            ) where {T<:Real}
        Construct a `WaterProperties` type.

        The constructor uses DIPPR105 to calculate the density at the given temperature.
        Parameters for the Quan-Fry parametrisation of the refractive index are calculated
        for the given salinity, temperature and pressure.
    """
    function WaterProperties(
        salinity::Unitful.Quantity{T},
        pressure::Unitful.Quantity{T},
        temperature::Unitful.Quantity{T},
        vol_conc_small_part::Unitful.Quantity{T},
        vol_conc_large_part::Unitful.Quantity{T},
        radiation_length::Unitful.Quantity{T},
        mean_scattering_angle::T
    ) where {T<:Real}
        salinity = ustrip(T, u"permille", salinity)
        temperature = ustrip(T, u"°C", temperature)
        pressure = ustrip(T, u"atm", pressure)
        quan_fry_params = _calc_quan_fry_params(salinity, temperature, pressure)
        density = DIPPR105(temperature + 273.15)

        new{T}(
            salinity,
            pressure,
            temperature,
            ustrip(T, u"ppm", vol_conc_small_part),
            ustrip(T, u"ppm", vol_conc_large_part),
            ustrip(T, u"g/cm^2", radiation_length),
            density,
            mean_scattering_angle,
            quan_fry_params
        )
    end
end

"""
    make_cascadia_medium_properties(::Type{T}) where {T<:Real}
Construct `WaterProperties` with properties from Cascadia Basin of numerical type `T`.
"""
make_cascadia_medium_properties(mean_scattering_angle::T) where {T<:Real} = WaterProperties(
    T(34.82)u"permille",
    T(269.44088)u"bar",
    T(1.8)u"°C",
    T(0.0075)u"ppm",
    T(0.0075)u"ppm",
    T(36.08)u"g/cm^2",
    mean_scattering_angle)



salinity(::T) where {T<:MediumProperties} = error("Not implemented for $T")
salinity(x::WaterProperties) = x.salinity

pressure(::T) where {T<:MediumProperties} = error("Not implemented for $T")
pressure(x::WaterProperties) = x.pressure

temperature(::T) where {T<:MediumProperties} = error("Not implemented for $T")
temperature(x::WaterProperties) = x.temperature

material_density(::T) where {T<:MediumProperties} = error("Not implemented for $T")
material_density(x::WaterProperties) = x.density

vol_conc_small_part(::T) where {T<:MediumProperties} = error("Not implemented for $T")
vol_conc_small_part(x::WaterProperties) = x.vol_conc_small_part


vol_conc_large_part(::T) where {T<:MediumProperties} = error("Not implemented for $T")
vol_conc_large_part(x::WaterProperties) = x.vol_conc_large_part

radiation_length(::T) where {T<:MediumProperties} = error("Not implemented for $T")
radiation_length(x::WaterProperties) = x.radiation_length

mean_scattering_angle(::T) where {T<:MediumProperties} = error("Not implemented for $T")
mean_scattering_angle(x::WaterProperties) = x.mean_scattering_angle


"""
    _calc_quan_fry_params(salinity::Real, temperature::Real, pressure::Real)

Helper function to get the parameters for the Quan & Fry formula as function of
salinity, temperature and pressure.
"""
function _calc_quan_fry_params(
    salinity::Real,
    temperature::Real,
    pressure::Real)

    n0 = 1.31405
    n1 = 1.45e-5
    n2 = 1.779e-4
    n3 = 1.05e-6
    n4 = 1.6e-8
    n5 = 2.02e-6
    n6 = 15.868
    n7 = 0.01155
    n8 = 0.00423
    n9 = 4382
    n10 = 1.1455e6

    a01 = (
        n0
        +
        (n2 - n3 * temperature + n4 * temperature^2) * salinity
        -
        n5 * temperature^2
        +
        n1 * pressure
    )
    a2 = n6 + n7 * salinity - n8 * temperature
    a3 = -n9
    a4 = n10

    return a01, a2, a3, a4
end

"""
    refractive_index_fry(wavelength, salinity, temperature, pressure)

The phase refractive index of sea water according to a model
from Quan & Fry.

wavelength is given in nm, salinity in permille, temperature in °C and pressure in atm

The original model is taken from:
X. Quan, E.S. Fry, Appl. Opt., 34, 18 (1995) 3477-3480.

An additional term describing pressure dependence was included according to:
Wolfgang H.W.A. Schuster, "Measurement of the Optical Properties of the Deep
Mediterranean - the ANTARES Detector Medium.",
PhD thesis (2002), St. Catherine's College, Oxford
downloaded Jan 2011 from: http://www.physics.ox.ac.uk/Users/schuster/thesis0098mmjhuyynh/thesis.ps

Adapted from clsim (©Claudio Kopper)
"""
function refractive_index_fry(
    wavelength::T;
    salinity::Real,
    temperature::Real,
    pressure::Real) where {T<:Real}
    refractive_index_fry(wavelength, T.(_calc_quan_fry_params(salinity, temperature, pressure)))
end

function refractive_index_fry(
    wavelength::Real,
    quan_fry_params::Tuple{U, U, U, U}
) where {U<:Real}

    a01, a2, a3, a4 = quan_fry_params
    x = one(wavelength) / wavelength
    x2 = x * x
    # a01 + x*a2 + x^2 * a3 + x^3 * a4
    return oftype(wavelength, fma(x, a2, a01) + fma(x2, a3, x2*x*a4))
end

function refractive_index_fry(
    wavelength::Unitful.Length{T};
    salinity::Unitful.DimensionlessQuantity,
    temperature::Unitful.Temperature,
    pressure::Unitful.Pressure) where {T<:Real}

    refractive_index_fry(
        ustrip(T, u"nm", wavelength),
        salinity=ustrip(u"permille", salinity),
        temperature=ustrip(u"°C", temperature),
        pressure=ustrip(u"atm", pressure)
    )
end

"""
    refractive_index(wavelength, medium)

Return the refractive index at `wavelength` for `medium`
"""
refractive_index(wavelength::Real, medium::WaterProperties) = refractive_index_fry(
    wavelength,
    medium.quan_fry_params
)

refractive_index(wavelength::Unitful.Length, medium::MediumProperties) = refractive_index(
    ustrip(u"nm", wavelength),
    medium)


"""
dispersion_fry(
    wavelength::T;
    salinity::Real,
    temperature::Real,
    pressure::Real) where {T <: Real}

    Calculate the dispersion (dn/dλ) for the Quan & Fry model.
    wavelength is given in nm, salinity in permille, temperature in °C and pressure in atm
"""
function dispersion_fry(
    wavelength::T;
    salinity::Real,
    temperature::Real,
    pressure::Real) where {T <: Real}
    dispersion_fry(wavelength, T.(_calc_quan_fry_params(salinity, temperature, pressure)))
end

function dispersion_fry(wavelength::T, quan_fry_params::Tuple{T, T, T, T}) where {T<:Real}
    a2, a3, a4 = quan_fry_params
    x = one(T) / wavelength

    return T(a2 + T(2)*x*a3 + T(3)*x^2*a4) * T(-1)/wavelength^2
end

dispersion(wavelength::Real, medium::WaterProperties) = dispersion_fry(
    wavelength,
    medium.quan_fry_params
)

"""
    cherenkov_angle(wavelength, medium::MediumProperties)
Calculate the cherenkov angle (in rad) for `wavelength` and `medium`.
"""
function cherenkov_angle(wavelength, medium::MediumProperties)
    return acos(one(typeof(wavelength))/refractive_index(wavelength, medium))
end

function group_velocity(wavelength::T, medium::MediumProperties) where {T<:Real}
    global c_vac_m_ns
    ref_ix::T = refractive_index(wavelength, medium)
    λ_0::T = ref_ix * wavelength
    T(c_vac_m_ns) / (ref_ix - λ_0 * dispersion(wavelength, medium))
end


"""
    _sca_len_part_conc(wavelength; vol_conc_small_part, vol_conc_large_part)

Calculates the scattering length (in m) for a given wavelength based on concentrations of
small (`vol_conc_small_part`) and large (`vol_conc_large_part`) particles.
wavelength is given in nm, vol_conc_small_part and vol_conc_large_part in ppm


Adapted from clsim ©Claudio Kopper
"""
@inline function _sca_len_part_conc(
    wavelength::T;
    vol_conc_small_part::Real,
    vol_conc_large_part::Real) where {T<:Real}

    ref_wlen::T = 550  # nm
    x::T = ref_wlen / wavelength

    sca_coeff = (
        T(0.0017) * x^T(4.3)
        + T(1.34) * vol_conc_small_part * x^T(1.7)
        + T(0.312) * vol_conc_large_part * x^T(0.3)
    )

    return T(1 / sca_coeff)

end

function _sca_len_part_conc(
    wavelength::Unitful.Length;
    vol_conc_small_part::Unitful.DimensionlessQuantity,
    vol_conc_large_part::Unitful.DimensionlessQuantity)

    _sca_len_part_conc(
        ustrip(u"nm", wavelength),
        vol_conc_small_part=ustrip(u"ppm", vol_conc_small_part),
        vol_conc_large_part=ustrip(u"ppm", vol_conc_large_part))
end

"""
    scattering_length(wavelength, medium)

Return the scattering length for a given wavelength and medium
"""
@inline function scattering_length(wavelength::Real, medium::WaterProperties)
    _sca_len_part_conc(
        wavelength;
        vol_conc_small_part=vol_conc_small_part(medium),
        vol_conc_large_part=vol_conc_large_part(medium))
end


function scattering_length(wavelength::Unitful.Length, medium::MediumProperties)
    scattering_length(ustrip(u"nm", wavelength), medium)
end


"""
    _absorption_length_straw(wavelength::Real)
Calculate the absorption length at `wavelength` (in nm).

Based on interpolation of STRAW attenuation length measurement.
"""
function _absorption_length_straw(wavelength::Real)
    T = typeof(wavelength)
    x = [T(300.0), T(365.0), T(400.0), T(450.0), T(585.0), T(800.0)]
    y = [T(10.4), T(10.4), T(14.5), T(27.7), T(7.1), T(7.1)]

    fast_linear_interp(wavelength, x, y)
end

struct AbsLengthStrawFromFit
    df::DataFrame
end

const ABSLENGTHSTRAWFIT = AbsLengthStrawFromFit(
    DataFrame(read_parquet(joinpath(@__DIR__, "../../assets/attenuation_length_straw_fit.parquet"))))

function (f::AbsLengthStrawFromFit)(wavelength::Real)
    T = typeof(wavelength)
    x::Vector{T} = f.df[:, :wavelength]
    y::Vector{T} = f.df[:, :abs_len]
    fast_linear_interp(wavelength, x, y)
end


"""
    absorption_length(wavelength::T, ::WaterProperties) where {T<:Real}

Return the absorption length (in m) for `wavelength` (in nm)
"""
function absorption_length(wavelength::T, ::WaterProperties) where {T<:Real}
    return ABSLENGTHSTRAWFIT(wavelength)
end


end # Module
