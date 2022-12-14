module PMTPipeline

using DataFrames
using DSP
using PoissonRandom
using Distributions
import Base: @kwdef
import Pipe: @pipe
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
export plot_hits, plot_pmt_map, map_f_over_pmts



function calc_gamma_shape_mean_fwhm(mean, target_fwhm)
    function _optim(theta)
        alpha = mean / theta
        tt_dist = Gamma(alpha, theta)
        fwhm(tt_dist, mode(tt_dist); xlims=(0, 100)) - target_fwhm
    end

    find_zero(_optim, [0.1 * target_fwhm^2 / mean, 10 * target_fwhm^2 / mean], A42())
end


@kwdef struct PMTConfig{T<:Real,S<:SPEDistribution{T},P<:PulseTemplate,U<:PulseTemplate,V<:UnivariateDistribution}
    spe_template::S
    pulse_model::P
    pulse_model_filt::U
    noise_amp::T
    sampling_freq::T # Ghz
    unf_pulse_res::T # ns
    adc_freq::T # Ghz
    tt_dist::V
    lp_filter::ZeroPoleGain{:z,ComplexF64,ComplexF64,Float64}

end

function PMTConfig(st::SPEDistribution, pm::PulseTemplate, snr_db::Real, sampling_freq::Real, unf_pulse_res::Real, adc_freq::Real, lp_cutoff::Real,
    tt_mean::Real, tt_fwhm::Real)
    mode = get_template_mode(pm)
    designmethod = Butterworth(1)
    lp_filter = digitalfilter(Lowpass(lp_cutoff, fs=sampling_freq), designmethod)
    filtered_pulse = make_filtered_pulse(pm, sampling_freq, (-10.0, 50.0), lp_filter)

    tt_theta = calc_gamma_shape_mean_fwhm(tt_mean, tt_fwhm)
    tt_alpha = tt_mean / tt_theta
    tt_dist = Gamma(tt_alpha, tt_theta)


    PMTConfig(st, pm, filtered_pulse, mode / 10^(snr_db / 10), sampling_freq, unf_pulse_res, adc_freq, tt_dist, lp_filter)
end


STD_PMT_CONFIG = PMTConfig(
    ExponTruncNormalSPE(expon_rate=1.0, norm_sigma=0.3, norm_mu=1.0, trunc_low=0.0, peak_to_valley=3.1),
    PDFPulseTemplate(
        dist=truncated(Gumbel(0, gumbel_width_from_fwhm(5.0)) + 4, 0, 20),
        amplitude=1.0 #ustrip(u"A", 5E6 * ElementaryCharge / 20u"ns")
    ),
    20,
    2.0,
    0.1,
    0.25,
    0.125,
    25, # TT mean
    1.5 # TT FWHM
)

function resample_simulation(hit_times, total_weights, downsample=1.0)
    wsum = sum(total_weights)

    mask = total_weights .> 0
    hit_times = hit_times[mask]
    total_weights = total_weights[mask]

    norm_weights = ProbabilityWeights(copy(total_weights), wsum)
    nhits = min(pois_rand(wsum * downsample), length(hit_times))
    try
        sample(hit_times, norm_weights, nhits; replace=false)
    catch e
        @show length(hit_times)
        error("error")
    end
end


function resample_simulation(df::AbstractDataFrame; downsample=1.0, per_pmt=true, time_col=:time)


    wrapped(hit_times, total_weights) = resample_simulation(hit_times, total_weights, downsample)

    if per_pmt
        groups = groupby(df, [:pmt_id, :module_id])
    else
        groups = groupby(df, :module_id)
    end
    resampled_hits = combine(groups, [time_col, :total_weight] => wrapped => time_col)
    return resampled_hits
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



function make_reco_pulses(results::AbstractDataFrame, pmt_config::PMTConfig=STD_PMT_CONFIG)
    @pipe results |>
          resample_simulation |>
          apply_tt!(_, pmt_config.tt_dist) |>
          subtract_mean_tt!(_, pmt_config.tt_dist) |>
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

function plot_hits(target, groups...; ylabel="", title="")
    l = grid(4, 4)
    plots = []

    coords = rad2deg.(target.pmt_coordinates)

    for (i, (theta, phi)) in enumerate(eachcol(coords))
        p = plot(title=format("??={:.2f}, ??={:.2f}", theta, phi), titlefontsize=8,)

        for grp in groups
            this_hits = get(grp, (pmt_id=i,), nothing)

            if !isnothing(this_hits)
                histogram!(p, this_hits[:, :time], bins=70:1:150, xlabel="Time (ns)", #yscale=:log10,
                    #ylim=(0.1, 5000),
                    label="",
                    yscale=:log10, ylim=(0.5, 1000),
                    alpha=0.7,
                    ylabel=ylabel,
                    margin=3.2Plots.mm, xlabelfontsize=8, ylabelfontsize=8,
                    legend_position=:outertopright,
                    #legend_columns=2,
                    legendfontsize=6,
                )
            end
        end

        push!(plots, p)

    end
    return plot(plots..., layout=l, size=(1200, 800), plot_title=title)
end

function plot_pmt_map(target, xmaps...; labels, ylabel="", title="")
    l = grid(4, 4)
    plots = []

    coords = rad2deg.(target.pmt_coordinates)
    first = true
    for (i, (theta, phi)) in enumerate(eachcol(coords))
        if first
            p = plot(title=format("??={:.2f}, ??={:.2f}", theta, phi), titlefontsize=8,
                legend_column=2,
                legendfontsize=6,
                legend_position=:best)
        else
            p = plot(title=format("??={:.2f}, ??={:.2f}", theta, phi), titlefontsize=8,
                legend_position=false)
        end
        first = false


        for (xmap, label) in zip(xmaps, labels)

            x = get(xmap, i, nothing)

            if isnothing(x)
                continue
            end
            plot!(p, x, label=label, ylabel=ylabel, xlabel="Time (ns)",
                margin=3.2Plots.mm, xlabelfontsize=8, ylabelfontsize=8,
                #legend_position=:outertopright,
            )
        end
        push!(plots, p)

    end
    return plot(plots..., layout=l, size=(1200, 800), plot_title=title)
end


function map_f_over_pmts(target, f, input)
    out_d = []
    for pmt_id in 1:get_pmt_count(target)
        if typeof(input) <: GroupedDataFrame
            in = get(input, (pmt_id=pmt_id,), nothing)
        else
            in = get(input, pmt_id, nothing)
        end
        if !isnothing(in)
            out = f(in)
            push!(out_d, (pmt_id, out))
        else
            push!(out_d, (pmt_id, nothing))
        end

    end

    return Dict(out_d)

end




end
