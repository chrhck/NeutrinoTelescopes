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



STD_PMT_CONFIG
spe_d = make_spe_dist(STD_PMT_CONFIG.spe_template)

plot(x -> pdf(spe_d, x), 0, 5, ylabel="PDF", xlabel="Charge (PE)",
    title="SPE Template")

plot(x -> evaluate_pulse_template(STD_PMT_CONFIG.pulse_model, 0.0, x), -50, 50, ylabel="Amplitude (a.u.)", xlabel="Time (ns)")


distance = 50f0
n_pmts=16
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)

targets = [target]

zenith_angle = 0f0
azimuth_angle = 0f0

pdir = sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle))

particle = Particle(
        @SVector[0.0f0, 0f0, 0.0f0],
        pdir,
        0f0,
        Float32(1E5),
        PEMinus
)

medium = make_cascadia_medium_properties(Float32)

prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
prop_source_che = PointlikeCherenkovEmitter(particle, medium, (300f0, 800f0))

results_che, nph_sim_che = propagate_source(prop_source_che, distance, medium)
results_ext, nph_sim_ext = propagate_source(prop_source_ext, distance, medium)

hit_times = resample_simulation(results_che)
histogram(hit_times, bins=-20:100)

hit_times = resample_simulation(results_ext)
histogram!(hit_times, bins=-20:100)

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