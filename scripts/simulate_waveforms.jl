using NeutrinoTelescopes
using Plots
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DSP
using Profile
using DataFrames
using BenchmarkTools
using StatsPlots
using GaussianProcesses
using Optim
import Pipe:@pipe
using PoissonRandom


spe_d = make_spe_dist(STD_PMT_CONFIG.spe_template)
plot(x -> pdf(spe_d, x), 0, 5, ylabel="PDF", xlabel="Charge (PE)",
    title="SPE Template")

plot(x -> evaluate_pulse_template(STD_PMT_CONFIG.pulse_model, 0.0, x), -50, 50, ylabel="Amplitude (a.u.)", xlabel="Time (ns)", label="Pulse template")
plot!(x -> evaluate_pulse_template(STD_PMT_CONFIG.pulse_model_filt, 0.0, x), -50, 50, label="Filtered")

distance = 50f0
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = MultiPMTDetector(@SVector[0.0f0, 0.0f0, distance], target_radius, pmt_area,
    make_pom_pmt_coordinates(Float32))
medium = make_cascadia_medium_properties(0.99f0)
targets = [target]


zenith_angle = 20f0
azimuth_angle = 0f0

particle = Particle(
        @SVector[0.0f0, 0f0, 0.0f0],
        sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
        0f0,
        Float32(1E5),
        PEMinus
)


prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
#prop_source_che = PointlikeCherenkovEmitter(particle, medium, (300f0, 800f0))

spectrum = CherenkovSpectrum((300f0, 800f0), 30, medium)

res = propagate_photons(prop_source_ext, target, medium, spectrum)

coszen_orient = 0
phi_orient = 0

orientation = sph_to_cart(acos(coszen_orient), phi_orient,)



res = make_hits_from_photons(res, prop_source_ext, target, medium, orientation)
res_grp_pmt = groupby(res, :pmt_id)

results = res_grp_pmt[5]
pmt_config = STD_PMT_CONFIG
waveform = @pipe results |>
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
      )



p = histogram(results[:, :time], bins=200:250, xlabel="Time (ns)",
          ylabel="Counts", label="")
savefig(p, joinpath(@__DIR__, "../figures/example_photons.png"))
reco_pulses = make_reco_pulses(results)
p = plot(waveform, xlabel="Time (ns)", xlim=(200, 250),
         ylabel="Amplitude (a.u.)", label="")#xlim=(-50, 200))
savefig(p, joinpath(@__DIR__, "../figures/example_waveform.png"))


function plot_chain(results::AbstractDataFrame, pmt_config::PMTConfig)
    layout = @layout [a; b]

    p1 = histogram(results[:, :tres], weights=results[:, :total_weight],
                   bins=-10:1:50, label="Photons", xlabel="Time residual (ns)", ylabel="Counts")

    hit_times = resample_simulation(results)
    p1 = histogram!(p1, hit_times, bins=-10:1:50, label="Hits")

    ps = PulseSeries(hit_times, pmt_config.spe_template, pmt_config.pulse_model)
    p2 = plot(ps, -10:0.01:50, label="True waveform", xlabel="Time residual (ns)", ylabel="Amplitude (a.u.)")

    wf = digitize_waveform(
        ps,
        pmt_config.sampling_freq,
        pmt_config.adc_freq,
        pmt_config.noise_amp,
        pmt_config.lp_filter
      )

    p2 = plot!(p2, wf, label="Digitized waveform")

    reco_pulses = unfold_waveform(wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.2, :fnnls)
    pulses_orig_temp = PulseSeries(reco_pulses.times, reco_pulses.charges, pmt_config.pulse_model)


    p2 = plot!(p2, reco_pulses, -10:0.01:50, label="Reconstructed waveform")
    p2 = plot!(p2, pulses_orig_temp, -10:0.01:50, label="Unfolded waveform")

    plot(p1, p2, layout=layout, xlim=(-10, 50))
end


plot_chain(results, STD_PMT_CONFIG)

reco_pulses_che.times

xs = -10:1.:50
ys = reco_pulses_che(xs)


mZero = MeanZero()                   #Zero mean function
kern = Mat32Iso(1.0, 1.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
gp = GP(xs,ys,mZero,kern,logObsNoise)

plot(reco_pulses_che)

plot!(gp, obsv=false, linestyle=:dash)

optimize!(gp)

plot!(gp, obsv=false,)


gp








hit_times = resample_simulation(results_che)
histogram(hit_times, bins=-20:5:100, alpha=0.7)





hit_times = resample_simulation(results_ext)
histogram!(hit_times, bins=-20:5:100, alpha=0.7)

#wf_model = make_gumbel_waveform_model(hit_times)
ps = PulseSeries(hit_times, STD_PMT_CONFIG.spe_template, STD_PMT_CONFIG.pulse_model)

plot(ps, -10, 200)


wf = make_waveform(ps, STD_PMT_CONFIG.sampling_freq, STD_PMT_CONFIG.noise_amp)



l = @layout [a; b]
p1 = plot(wf, label="Waveform + noise")
p2 = histogram(ps.times, weights=ps.charges, bins=-50:1:250, label="PE")
plot(p1, p2, layout=l)

designmethod = Butterworth(1)
lp_filter = digitalfilter(Lowpass(STD_PMT_CONFIG.lp_cutoff, fs=STD_PMT_CONFIG.sampling_freq), designmethod)

pulse_model_filt = make_filtered_pulse(STD_PMT_CONFIG.pulse_model, STD_PMT_CONFIG.sampling_freq, (-1000.0, 1000.0), lp_filter)

digi_wf = digitize_waveform(
    ps,
    STD_PMT_CONFIG.sampling_freq,
    STD_PMT_CONFIG.adc_freq,
    STD_PMT_CONFIG.noise_amp,
    lp_filter,
)

plot(wf, label="Waveform + Noise", xlabel="Time (ns)", ylabel="Amplitude (a.u.)", right_margin=40Plots.px,
    xlim=(-20, 50))
plot!(digi_wf, label="Digitized Waveform")
#plot!(ps, -20, 50)
sticks!(twinx(), ps.times, ps.charges, legend=false, left_margin=30Plots.px, ylabel="Charge (PE)", ylim=(1, 20), color=:green, xticks=:none, xlim=(-20, 50))


reco_pulses = unfold_waveform(digi_wf, pulse_model_filt, STD_PMT_CONFIG.unf_pulse_res, 0.2, :fnnls)

plot_waveform(wf, digi_wf, reco_pulses, STD_PMT_CONFIG.pulse_model, pulse_model_filt, (0.0, maximum(wf.values) * 1.1))





#=





data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
model = data[:model] |> gpu

output_trafos = [:log, :log, :neg_log_scale]

positions = make_detector_cube(5, 5, 10, 50.0, 100.0)
targets = make_targets(positions)

zenith_angle = 25
azimuth_angle = 70




medium64 = make_cascadia_medium_properties(Float64)


model_params, sources = evaluate_model(targets, particle, medium64, 0.5, model, output_trafos)


poissons = poisson_dist_per_module(model_params, sources)
shapes = shape_mixture_per_module(model_params, sources)

event = sample_event(poissons, shapes, sources)

hit_times = event[argmax([length(t) for t in event])+2]
=
