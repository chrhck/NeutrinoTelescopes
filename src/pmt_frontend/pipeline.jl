module Pipeline

using DataFrames
using DSP
using PoissonRandom
using Distributions
import Base:@kwdef
import Pipe:@pipe
using PhysicalConstants.CODATA2018
using Unitful


using ..SPETemplates
using ..PulseTemplates
using ..Waveforms


export resample_simulation
export STD_PMT_CONFIG, PMTConfig
export make_reco_pulses



@kwdef struct PMTConfig{ T<:Real, S <: SPEDistribution{T}, P<:PulseTemplate, U<:PulseTemplate}
    spe_template::S
    pulse_model::P
    pulse_model_filt::U
    noise_amp::T
    sampling_freq::T # Ghz
    unf_pulse_res::T # ns
    adc_freq::T # Ghz
    lp_filter::ZeroPoleGain{:z, ComplexF64, ComplexF64, Float64}

end

function PMTConfig(st::SPEDistribution, pm::PulseTemplate, snr_db::Real, sampling_freq::Real, unf_pulse_res::Real, adc_freq::Real, lp_cutoff::Real)
    mode = get_template_mode(pm)
    designmethod = Butterworth(1)
    lp_filter = digitalfilter(Lowpass(lp_cutoff, fs=sampling_freq), designmethod)
    filtered_pulse = make_filtered_pulse(pm, sampling_freq, (-1000.0, 1000.0), lp_filter)
    PMTConfig(st, pm, filtered_pulse, mode / 10^(snr_db / 10), sampling_freq, unf_pulse_res, adc_freq, lp_filter)
end



STD_PMT_CONFIG = PMTConfig(
    ExponTruncNormalSPE(expon_rate=1.0, norm_sigma=0.3, norm_mu=1.0, trunc_low=0.0, peak_to_valley=3.1),
    PDFPulseTemplate(
        dist=truncated(Gumbel(0, gumbel_width_from_fwhm(5.0))+4, 0, 20),
        amplitude=1. #ustrip(u"A", 5E6 * ElementaryCharge / 20u"ns")
    ),
    20,
    2.0,
    0.1,
    0.25,
    0.125
)

function resample_simulation(df::AbstractDataFrame )
    hit_times = df[:, :tres]
    wsum = sum(df[:, :total_weight])
    norm_weights = df[:, :total_weight] ./ wsum
    nhits = pois_rand(wsum)
    d = Categorical(norm_weights)
    hit_times = hit_times[rand(d, nhits)]
end


function make_reco_pulses(results::AbstractDataFrame , pmt_config::PMTConfig=STD_PMT_CONFIG)
    @pipe results |>
      resample_simulation |>
      PulseSeries(_, pmt_config.spe_template, pmt_config.pulse_model) |>
      digitize_waveform(
        _,
        pmt_config.sampling_freq,
        pmt_config.adc_freq,
        pmt_config.noise_amp,
        pmt_config.lp_filter
      ) |>
      unfold_waveform(_, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.2, :fnnls)
end
end