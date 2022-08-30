module Pipeline

using DataFrames
using DSP
using PoissonRandom
using Distributions
import Base:@kwdef
import Pipe:@pipe
using PhysicalConstants.CODATA2018
using Unitful
using StatsBase
using Random
using Interpolations
using Roots

using ..SPETemplates
using ..PulseTemplates
using ..Waveforms
using ...Utils


export resample_simulation
export STD_PMT_CONFIG, PMTConfig
export make_reco_pulses
export calc_gamma_shape_mean_fwhm
export apply_tt, apply_tt!, subtract_mean_tt, subtract_mean_tt!



function calc_gamma_shape_mean_fwhm(mean, target_fwhm)


    function _optim(theta)
        alpha = mean / theta
        tt_dist = Gamma(alpha, theta)
        fwhm(tt_dist, mode(tt_dist); xlims=(0, 100)) - target_fwhm
    end

    find_zero(_optim, [0.1*target_fwhm^2/mean, 10*target_fwhm^2/mean], A42())
end



@kwdef struct PMTConfig{ T<:Real, S <: SPEDistribution{T}, P<:PulseTemplate, U<:PulseTemplate, V<:UnivariateDistribution}
    spe_template::S
    pulse_model::P
    pulse_model_filt::U
    noise_amp::T
    sampling_freq::T # Ghz
    unf_pulse_res::T # ns
    adc_freq::T # Ghz
    tt_dist::V
    lp_filter::ZeroPoleGain{:z, ComplexF64, ComplexF64, Float64}

end

function PMTConfig(st::SPEDistribution, pm::PulseTemplate, snr_db::Real, sampling_freq::Real, unf_pulse_res::Real, adc_freq::Real, lp_cutoff::Real,
    tt_mean::Real, tt_fwhm::Real)
    mode = get_template_mode(pm)
    designmethod = Butterworth(1)
    lp_filter = digitalfilter(Lowpass(lp_cutoff, fs=sampling_freq), designmethod)
    filtered_pulse = make_filtered_pulse(pm, sampling_freq, (-10.0, 50.), lp_filter)

    tt_theta = calc_gamma_shape_mean_fwhm(tt_mean, tt_fwhm)
    tt_alpha = tt_mean / tt_theta
    tt_dist = Gamma(tt_alpha, tt_theta)


    PMTConfig(st, pm, filtered_pulse, mode / 10^(snr_db / 10), sampling_freq, unf_pulse_res, adc_freq, tt_dist, lp_filter)
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
    0.125,
    25, # TT mean
    1.5 # TT FWHM
)

function resample_simulation(hit_times, total_weights, downsample=1.)
    wsum = sum(total_weights)
    norm_weights = ProbabilityWeights(copy(total_weights), wsum)
    nhits = pois_rand(wsum*downsample)
    sample(hit_times, norm_weights, nhits; replace=false)
end


function resample_simulation(df::AbstractDataFrame, downsample=1., per_pmt=true)
    
    
    wrapped(hit_times, total_weights) = resample_simulation(hit_times, total_weights, downsample)
    
    if per_pmt
        groups = groupby(df, :pmt_id)
        resampled_hits = combine(groups, [:time, :total_weight] => wrapped => :time)
    else
        resampled_hits = combine(df, [:time, :total_weight] => wrapped => :time)
    end
    resampled_hits
end



function apply_tt(hit_times::AbstractArray{<:Real}, tt_dist::UnivariateDistribution)

    tt = rand(tt_dist, size(hit_times))
    return hit_times .+ tt
end


function apply_tt!(df::AbstractDataFrame, tt_dist::UnivariateDistribution)
    tt = rand(tt_dist, nrow(df))

    df[!, :time] .+= tt
    return df
end


function subtract_mean_tt(hits::AbstractVector{<:Real}, tt_dist::UnivariateDistribution)
    hits .- mean(tt_dist)
end

function subtract_mean_tt!(df::AbstractDataFrame, tt_dist::UnivariateDistribution)
    df[!, :time] .-= mean(tt_dist)
    return df
end



function make_reco_pulses(results::AbstractDataFrame , pmt_config::PMTConfig=STD_PMT_CONFIG)
    @pipe results |>
      resample_simulation |>
      apply_tt(_, pmt_config.tt_dist) |>
      subtract_mean_tt(_, pmt_config.tt_dist) |>
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