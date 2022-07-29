module Waveforms

using ..PulseTemplates
using Plots
using NonNegLeastSquares
using NNLS
using DSP
export Waveform
export add_gaussian_white_noise, digitize_waveform, unfold_waveform, plot_waveform

struct Waveform{T<:Real,V<:AbstractVector{T}}
    timestamps::V
    values::V
end

@recipe function f(wf::T) where {T<:Waveform}
    x := wf.timestamps
    y := wf.values
    ()
end

function add_gaussian_white_noise(values, scale)
    values .+ randn(size(values)) * scale
end


function digitize_waveform(
    pulse_series::PulseSeries{T},
    sampling_frequency::T,
    digitizer_frequency::T,
    noise_amp::T,
    filter,
    eval_range::Tuple{T,T},
) where {T<:Real}

    dt = 1 / sampling_frequency # ns
    timesteps = range(eval_range[1], eval_range[2], step=dt)

    waveform_values = evaluate_pulse_series(timesteps, pulse_series)
    waveform_values_noise = add_gaussian_white_noise(waveform_values, noise_amp)

    waveform_filtered = filt(filter, waveform_values_noise)

    resampling_rate = digitizer_frequency / sampling_frequency
    new_interval = range(eval_range[1], eval_range[2], step=1 / digitizer_frequency)
    waveform_resampled = resample(waveform_filtered, resampling_rate)

    return Waveform(collect(new_interval), waveform_resampled)
end

function make_nnls_matrix(
    pulse_times::V,
    pulse_shape::PulseTemplate{T},
    timestamps::V) where {T<:Real,V<:AbstractVector{T}}

    nnls_matrix = zeros(T, size(timestamps, 1), size(pulse_times, 1))

    # dt = 1/sampling_frequency # ns
    # timestamps_hires = range(eval_range[1], eval_range[2], step=dt)


    for i in eachindex(pulse_times)
        nnls_matrix[:, i] = evaluate_pulse_template(
            pulse_shape, pulse_times[i], timestamps)
    end

    nnls_matrix

end


function apply_nnls(
    pulse_times::V,
    pulse_shape::PulseTemplate{T},
    digi_wf::Waveform{T,V};
    alg::Symbol=:nnls) where {T<:Real,V<:AbstractVector{T}}

    matrix = make_nnls_matrix(pulse_times, pulse_shape, digi_wf.timestamps)
    #charges = nonneg_lsq(matrix, digi_wf.values; alg=:nnls)[:, 1]

    if alg == :nnls_NNLS
        charges = nnls(matrix, digi_wf.values)
    else
        charges = nonneg_lsq(matrix, digi_wf.values, alg=alg)[:, 1]
    end
    charges
end


function unfold_waveform(
    digi_wf::Waveform{T,V},
    pulse_model::PulseTemplate{T},
    pulse_resolution::T,
    min_charge::T,
    alg::Symbol=:nnls
) where {T<:Real,V<:AbstractVector{T}}

    min_time, max_time = extrema(digi_wf.timestamps)
    pulse_times = collect(range(min_time, max_time, step=pulse_resolution))
    pulse_charges = apply_nnls(pulse_times, pulse_model, digi_wf, alg=alg)

    nonzero = pulse_charges .> min_charge

    return pulse_times[nonzero], pulse_charges[nonzero]

end

function plot_waveform(
    orig_waveform::Waveform{T,V},
    digitized_waveform::Waveform{T,V},
    pulse_times::V,
    pulse_charges::V,
    pulse_template::PulseTemplate{T},
    reco_pulse_template::PulseTemplate{T},
    ylim::Tuple{T,T}
) where {T<:Real,V<:AbstractVector{T}}

    reco_wf = PulseSeries(pulse_times, pulse_charges, reco_pulse_template)
    reco_wf_uf = PulseSeries(pulse_times, pulse_charges, pulse_template)

    p = plot(
        orig_waveform,
        label="Waveform + Noise",
        xlabel="Time (ns)",
        ylabel="Amplitude (a.u.)",
        right_margin=45Plots.px,
        legend=:topleft,
        lw=2,
        palette=:seaborn_colorblind,
        dpi=150,
        ylim=ylim
        #xlim=(-100, 100)
    )
    p = plot!(digitized_waveform, label="Digitized Waveform", lw=2)
    p = plot!(digitized_waveform.timestamps, reco_wf, label="Reconstructed Waveform", lw=2)

    p = plot!(orig_waveform.timestamps, reco_wf_uf, label="Unfolded Waveform", lw=2)

    p = sticks!(twinx(), pulse_times, pulse_charges, legend=false, left_margin=30Plots.px, ylabel="Charge (PE)", ylim=(0, 10), color=:red, xticks=:none)

    p
end

end