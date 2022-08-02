### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 882822ac-0d8e-11ed-3403-cfcd68ec92fc
begin
	using Pkg
	Pkg.activate("..")
	
	using NeutrinoTelescopes
	using NeutrinoTelescopes.Modelling
	using NeutrinoTelescopes.PMTFrontEnd.PulseTemplates
	using NeutrinoTelescopes.PMTFrontEnd.SPETemplates
	using NeutrinoTelescopes.PMTFrontEnd.Waveforms
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
	using PlutoUI
   	using Printf
	using StatsPlots
	using StatsBase
	using Enzyme
end

# ╔═╡ 4b33aa8c-9644-4938-a3e4-ce1c857bebb3


# ╔═╡ f3f610e5-5345-4ddc-b5ac-caa4807c744f
begin
	data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
	model = data[:model] |> gpu
	
	output_trafos = [:log, :log, :neg_log]
	
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
	
	hit_times = event[argmax([length(t) for t in event])+1]
	
end

# ╔═╡ 1b6f5040-f704-4156-a659-fb718002f896
md"""
- Exp. decay: $(@bind expon_decay Slider(0.1:0.1:2, default=1., show_value=true)) 
- Exp. weight: $(@bind expon_weight Slider(0.1:0.05:0.9, default=0.3, show_value=true))
- Gumbel fwhm (ns): $(@bind gumbel_fwhm Slider(1:0.5:30, default=10, show_value=true))
- Amplitude: $(@bind pulse_amplitude Slider(10.:20:150., default=100., show_value=true))
- Sampling frequency (GHz): $(@bind sampling_freq Slider(1:0.2:3, default=2, show_value=true)) 
- SNRdb: $(@bind snr_db Slider(-10:1:30, default=10, show_value=true))
- ADC frequency (GHz): $(@bind adc_freq Slider(100E-3:10E-3:500E-3, default=200E-3, show_value=true))
- Lowpass cutoff (GHz): $(@bind lp_cutoff Slider(50E-3:10E-3:500E-3, default=125E-3, show_value=true))
- Initial pulse resolution (ns): $(@bind unf_pulse_res Slider(0.1:0.1:10, default=0.1, show_value=true))

- Min Charge: $(@bind min_charge Slider(0:0.05:1, default=0.1, show_value=true))

"""

# ╔═╡ f0f4c1aa-e4c9-41c3-bda9-02c98954a1c5
begin
	Random.seed!(31337)
	spe = ExponTruncNormalSPE(expon_decay, 0.3, 1.0, 0.0, expon_weight)
	spe_d = make_spe_dist(spe)
	gumbel_width = gumbel_width_from_fwhm(gumbel_fwhm)
	pulse_model = GumbelPulse(gumbel_width, pulse_amplitude)
	pulse = make_pulse_dist(pulse_model)
	gumbel_mode = evaluate_pulse_template(pulse_model, 0.0, [0.0])[1]
	
	snr = 10^(snr_db / 10)
	noise_amp = gumbel_mode / snr

	charges = rand(spe_d, size(hit_times))
	wf = PulseSeries(hit_times, charges, pulse_model)

	min_time, max_time = extrema(hit_times)
	min_time -= 35
	max_time += 35
	
	dt = 1 / sampling_freq # ns
	timestamps_hires = range(min_time, max_time, step=dt)
	
	waveform_values = evaluate_pulse_series(timestamps_hires, wf)
	if noise_amp > 0
		waveform_values_noise = add_gaussian_white_noise(waveform_values, noise_amp)
	else
		waveform_values_noise = waveform_values
	end

	designmethod = Butterworth(5)
	nyquist = adc_freq / 2
	lp_filter = digitalfilter(Lowpass(1*nyquist, fs=sampling_freq), designmethod)

	#lp_filter = digitalfilter(Lowpass(0.99), designmethod)
	
	pulse_model_filt = make_filtered_pulse(pulse_model, sampling_freq, (-1000.0, 1000.0), lp_filter)
	
	digi_wf = digitize_waveform(
	    wf,
	    sampling_freq,
	    adc_freq,
	    noise_amp,
	    lp_filter,
	    (min_time, max_time)
	)

	pulse_times, pulse_charges = unfold_waveform(digi_wf, pulse_model_filt, unf_pulse_res, min_charge, :nnls)

	orig_waveform = Waveform(collect(timestamps_hires), waveform_values_noise)

	reco_wf = PulseSeries(pulse_times, pulse_charges, pulse_model_filt)
    reco_wf_uf = PulseSeries(pulse_times, pulse_charges, pulse_model)

	reco_wf_uf_values = evaluate_pulse_series(timestamps_hires, reco_wf_uf)
	reco_wf_uf_values_filtered = filt(lp_filter, reco_wf_uf_values)

	
	l = @layout [a; b]
	#p1 = plot(timestamps_hires, waveform_values, label="Waveform", lw=2)
	p1 = plot(timestamps_hires, waveform_values_noise, label="Waveform + Noise", lw=2, ylabel="Amplitude (a.u.)")
	plot!(p1, digi_wf.timestamps, digi_wf.values, label="Digitized Waveform", lw=2)
	scatter!(p1, digi_wf.timestamps, reco_wf, label="Reconstructed Waveform", lw=2, ls=:dash)
	plot!(p1, timestamps_hires, reco_wf_uf, label="Refolded Waveform")
	plot!(p1, timestamps_hires, reco_wf_uf_values_filtered, label="Refolded Waveform LP", ls=:dash)
	
	p2 = histogram(hit_times, weights=charges, bins=0:1:100, label="Photons", ylabel="Counts")
	histogram!(p2, pulse_times, weights=pulse_charges, bins=0:1:100, label="Pulses")
	plot(p1, p2, layout=l, xlim=(-5, 120), xlabel="Time (ns)")

	

	
	
end

# ╔═╡ 7b4e0874-2a94-49e0-aecd-deccf18548e7
begin
	using FFTW
	function do_fft(signal, sampling_freq)
	
		L = length(signal)
		fft_vals = abs.(rfft(signal)) ./ L
		fft_vals_one = fft_vals[1:Int64(floor(L/2))+1]
		fft_vals_one[2:end-1] = 2*fft_vals_one[2:end-1]
		rfftfreq(L, sampling_freq), abs.(fft_vals_one)
	end

	freq1, fft1 = do_fft(reco_wf_uf_values, sampling_freq)
	freq2, fft2 = do_fft(reco_wf_uf_values_filtered, sampling_freq)

	plot(freq1, fft1, xlim=(0, 0.150), label="Refolded", xlabel="Frequency Ghz")
	plot!(freq2, fft2, label="Refolded filtered")
	vline!([nyquist], label="")

end
	

# ╔═╡ 0a833a22-a97f-42b3-99a0-7e953f683216
begin
	p = plot()
	for n in 1:5
		lp_filt = digitalfilter(Lowpass(nyquist), Butterworth(n))
		H, w = freqresp(lp_filt)
		plot!(w * adc_freq, abs.(H))
	end
p
end

# ╔═╡ 171d0f08-e62c-4bdf-aea2-4fc5c3f911fe
plot(timestamps_hires, reco_wf_uf_values- reco_wf_uf_values_filtered)

# ╔═╡ Cell order:
# ╠═882822ac-0d8e-11ed-3403-cfcd68ec92fc
# ╠═4b33aa8c-9644-4938-a3e4-ce1c857bebb3
# ╠═f3f610e5-5345-4ddc-b5ac-caa4807c744f
# ╠═1b6f5040-f704-4156-a659-fb718002f896
# ╠═f0f4c1aa-e4c9-41c3-bda9-02c98954a1c5
# ╠═0a833a22-a97f-42b3-99a0-7e953f683216
# ╠═7b4e0874-2a94-49e0-aecd-deccf18548e7
# ╠═171d0f08-e62c-4bdf-aea2-4fc5c3f911fe
