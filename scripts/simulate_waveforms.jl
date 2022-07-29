using Revise
using NeutrinoTelescopes
using NeutrinoTelescopes.Modelling
using NeutrinoTelescopes.PMTFrontEnd
using NeutrinoTelescopes.Utils
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.Detection
using Plots
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DSP

expon_decay = 1.0
expon_weight = 0.3
gumbel_fwhm = 10.0
pulse_amplitude = 100.0
snr_db = 10
sampling_freq = 2.0
unf_pulse_res = 0.1
adc_freq = 200E-3
lp_cutoff = 125E-3

spe = PMTFrontEnd.ExponTruncNormalSPE(expon_decay, 0.3, 1.0, 0.0, expon_weight)
spe_d = PMTFrontEnd.make_spe_dist(spe)


gumbel_width = PMTFrontEnd.gumbel_width_from_fwhm(gumbel_fwhm)
pulse_model = PMTFrontEnd.GumbelPulse(gumbel_width, pulse_amplitude)
pulse = PMTFrontEnd.make_pulse_dist(pulse_model)
gumbel_mode = PMTFrontEnd.evaluate_pulse_template(pulse_model, 0.0, [0.0])[1]


plot(x -> pdf(spe_d, x), 0, 5, ylabel="PDF", xlabel="Charge (PE)",
    title="Exp decay: $expon_decay, Exp weight: $expon_weight")

plot(x -> PMTFrontEnd.evaluate_pulse_template(pulse_model, 0.0, x), -50, 50, ylabel="Amplitude (a.u.)", xlabel="Time (ns)")



snr = 10^(snr_db / 10)
noise_amp = gumbel_mode / snr

data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
model = data[:model] |> gpu

output_trafos = [:log, :log, :neg_log_scale]

positions = make_detector_cube(5, 5, 10, 50.0, 100.0)
targets = make_targets(positions)

zenith_angle = 25
azimuth_angle = 70


particle = LightYield.Particle(
    @SVector[0.0, 0.0, 20.0],
    sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
    0.0,
    1E5,
    LightYield.EMinus
)

medium64 = make_cascadia_medium_properties(Float64)


model_params, sources = evaluate_model(targets, particle, medium64, 0.5, model, output_trafos)


poissons = poisson_dist_per_module(model_params, sources)
shapes = shape_mixture_per_module(model_params, sources)

event = sample_event(poissons, shapes, sources)

hit_times = event[argmax([length(t) for t in event])+2]

charges = rand(spe_d, size(hit_times))


#wf_model = make_gumbel_waveform_model(hit_times)
wf = PMTFrontEnd.PulseSeries(hit_times, charges, pulse_model)

plot(wf, 0, 200)

min_time, max_time = extrema(hit_times)
min_time -= 30
max_time += 30

dt = 1 / sampling_freq # ns
timestamps_hires = range(min_time, max_time, step=dt)

waveform_values = PMTFrontEnd.evaluate_pulse_series(timestamps_hires, wf)
if noise_amp > 0
    waveform_values_noise = PMTFrontEnd.add_gaussian_white_noise(waveform_values, noise_amp)
else
    waveform_values_noise = waveform_values
end


l = @layout [a; b]
p1 = plot(timestamps_hires, waveform_values, label="Waveform")
p1 = plot!(timestamps_hires, waveform_values_noise, label="Waveform + Noise")

p2 = histogram(hit_times, weights=charges, bins=0:1:100, label="Photons")
plot(p1, p2, layout=l)

designmethod = Butterworth(1)
lp_filter = digitalfilter(Lowpass(lp_cutoff, fs=sampling_freq), designmethod)

pulse_model_filt = PMTFrontEnd.make_filtered_pulse(pulse_model, sampling_freq, (-1000.0, 1000.0), lp_filter)

digi_wf = PMTFrontEnd.digitize_waveform(
    wf,
    sampling_freq,
    adc_freq,
    noise_amp,
    lp_filter,
    (min_time, max_time)
)

plot(timestamps_hires, waveform_values_noise, label="Waveform + Noise", xlabel="Time (ns)", ylabel="Amplitude (a.u.)", right_margin=40Plots.px,
    xlim=(-20, 250))
plot!(digi_wf.timestamps, digi_wf.values, label="Digitized Waveform")
sticks!(twinx(), hit_times, charges, legend=false, left_margin=30Plots.px, ylabel="Charge (PE)", ylim=(1, 20), color=:green, xticks=:none, xlim=(-20, 50))


pulse_times, pulse_charges = PMTFrontEnd.unfold_waveform(digi_wf, pulse_model_filt, unf_pulse_res, 0.2, :fnnls)

orig_waveform = PMTFrontEnd.Waveform(collect(timestamps_hires), waveform_values_noise)

PMTFrontEnd.plot_waveform(orig_waveform, digi_wf, pulse_times, pulse_charges, pulse_model, pulse_model_filt, (0.0, maximum(orig_waveform.values) * 1.1))