module PulseTemplates

using Plots
using Polynomials
using Distributions
using Roots
using DSP
using Interpolations


export PulseTemplate, PDFPulseTemplate, GumbelPulse, InterpolatedPulse
export make_pulse_dist, evaluate_pulse_template, make_filtered_pulse
export PulseSeries, evaluate_pulse_series
export gumbel_width_from_fwhm
"""
Abstract type for pulse templates
"""
abstract type PulseTemplate{T<:Real} end

"""
Abstract type for pulse templates that use a PDF to define the pulse shape
"""
abstract type PDFPulseTemplate{T} <: PulseTemplate{T} end

"""
Pulse template using the gumbel-pdf to define its shape
"""
struct GumbelPulse{T} <: PDFPulseTemplate{T}
    sigma::T
    amplitude::T
end


"""
Pulse template using an interpolation to define its shape
"""
struct InterpolatedPulse{T} <: PulseTemplate{T}
    interp
    amplitude::T
end


"""
    fit_gumbel_fwhm_width()

Fit a polynomial to the relationship between Gumbel width and FWHM
"""
function fit_gumbel_fwhm_width()
    # find relationship between Gumbel width and FWHM
    function fwhm(d, xmode)
        ymode = pdf(d, xmode)

        z0 = find_zero(x -> pdf(d, x) - ymode / 2, (-20, xmode), Bisection())
        z1 = find_zero(x -> pdf(d, x) - ymode / 2, (xmode, 20), Bisection())
        return z1 - z0
    end

    widths = 0.5:0.01:5

    # Fit the function width = a * fwhm + b
    poly = Polynomials.fit(map(w -> fwhm(Gumbel(0, w), w), widths), widths, 1)
    poly
end

gumbel_width_from_fwhm = fit_gumbel_fwhm_width()


"""
    make_pulse_dist(p::PulseTemplate)

Return a `Distribution`
"""
make_pulse_dist(::T) where {T<:PulseTemplate} = error("not implemented")
make_pulse_dist(p::GumbelPulse{T}) where {T} = mu -> Gumbel(mu, p.sigma)

"""
    evaluate_pulse_template(pulse_shape::PulseTemplate, pulse_time::T, timestamp::T)

Evaluate a pulse template `pulse_shape` placed at time `pulse_time` at time `timestamp`
"""
evaluate_pulse_template(
    pulse_shape::PulseTemplate{T},
    pulse_time::T,
    timestamp::T) where {T} = error("not implemented")

function evaluate_pulse_template(
    pulse_shape::PDFPulseTemplate{T},
    pulse_time::T,
    timestamp::T) where {T}

    dist = make_pulse_dist(pulse_shape)(pulse_time)
    return pdf(dist, timestamp) * pulse_shape.amplitude
end

function evaluate_pulse_template(
    pulse_shape::InterpolatedPulse{T},
    pulse_time::T,
    timestamp::T) where {T}

    shifted_time = timestamp - pulse_time
    return pulse_shape.interp(shifted_time) * pulse_shape.amplitude

end

function evaluate_pulse_template(
    pulse_shape::PulseTemplate{T},
    pulse_time::T,
    timestamps::V) where {T,V<:AbstractVector{T}}

    evaluate_pulse_template.(Ref(pulse_shape), Ref(pulse_time), timestamps)
end

"""
    make_filteres_pulse(orig_pulse, sampling_frequency, eval_range, filter)

    Create filtered response of `orig_pulse` using `filter` and return
    `InterpolatedPulse`.
"""
function make_filtered_pulse(
    orig_pulse::PulseTemplate{T},
    sampling_freq::T,
    eval_range::Tuple{T,T},
    filter) where {T}

    timesteps = range(eval_range[1], eval_range[2], step=1 / sampling_freq)
    orig_eval = evaluate_pulse_template(orig_pulse, 0.0, timesteps)
    filtered = filt(filter, orig_eval)
    interp_linear = LinearInterpolation(timesteps, filtered)

    return InterpolatedPulse(interp_linear, 1.0)
end

struct PulseSeries{T<:Real,V<:AbstractVector{T}}
    times::V
    charges::V
    pulse_shape::PulseTemplate{T}

    function PulseSeries(
        times::V,
        charges::V,
        shape::PulseTemplate{T}) where {T<:Real,V<:AbstractVector{T}}

        ix = sortperm(times)
        return new{T,V}(times[ix], charges[ix], shape)
    end
end

function Base.:+(a::PulseSeries{T,V}, b::PulseSeries{T,V}) where {T<:Real,V<:AbstractVector{T}}
    # Could instead parametrize PulseSeries by PulseShape
    if a.pulse_shape != b.pulse_shape
        throw(ArgumentError("Pulse shapes are not compatible"))
    end
    PulseSeries([a.times; b.times], [a.charges; b.charges], a.pulse_shape)
end


function evaluate_pulse_series(time::T, wf::PulseSeries{T}) where {T<:Real}

    evaluated_wf = T(0)

    #for (ptime, pcharge) in zip(wf.times, wf.charges)	
    for i in 1:size(wf.times, 1)
        ptime = wf.times[i]
        pcharge = wf.charges[i]

        evaluated_wf += evaluate_pulse_template(wf.pulse_shape, ptime, time) * pcharge
    end
    evaluated_wf

end

function evaluate_pulse_series(times::V, wf::PulseSeries{T}) where {T<:Real,V<:AbstractVector{T}}
    evaluate_pulse_series.(times, Ref(wf))
end

@recipe function f(::Type{T}, ps::T) where {T<:PulseSeries}
    xi -> evaluate_pulse_series(xi, ps)
end

end