### A Pluto.jl notebook ###
# v0.19.10

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

# ╔═╡ c8f490da-d689-4f2b-8dcb-497c05973157
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
end

# ╔═╡ ffd29632-7e6e-4925-b7e2-3814ee23ebb6
md"""
# Waveform Digitization

This notebooks demonstrates PMT waveform digitization and unfolding
The steps are:

1) Convert photons to photo-electrons using SPE template, resulting in pulse times and charges

2) Convert photo-electrons into waveform using pulse template (linear combination)
3) Evaluate waveform at some high sampling rate
4) Lowpass-filter and resample waveform to ADC sampling frequency
5) Use NNLS to unfold digitized waveform into reconstructed pulses
"""

# ╔═╡ 3d7f7651-a64d-416c-9ea9-7455520fa7d7
md"""
## Parameters for the SPE Template

Exp. decay: $(@bind expon_decay Slider(0.1:0.1:2, default=1., show_value=true)) 

Exp. weight: $(@bind expon_weight Slider(0.1:0.05:0.9, default=0.3, show_value=true))
"""

# ╔═╡ c45eec73-54a0-463c-a752-2e7bccb82310
begin
	spe = ExponTruncNormalSPE(expon_decay, 0.3, 1., 0., expon_weight)
	spe_d = make_spe_dist(spe)
end

# ╔═╡ 15f8b0e3-247b-4e91-b5d9-a01ef7a5e29b
plot(x -> pdf(spe_d, x), 0, 5, ylabel="PDF", xlabel="Charge (PE)",
     title="Exp decay: $expon_decay, Exp weight: $expon_weight")

# ╔═╡ 1adb8eb9-dd29-4eb5-af04-9d62b044b9ba
md"""
# Parameters for the pulse template.
"""


# ╔═╡ f20109ca-878e-4407-88b2-21d31ea46ffb
md"""
Gumbel fwhm (ns): $(@bind gumbel_fwhm Slider(1:0.5:30, default=10, show_value=true))

Amplitude: $(@bind pulse_amplitude Slider(10.:20:150., default=100., show_value=true))
"""

# ╔═╡ 59635b6c-31ad-4749-bf71-8e55254fdbe3
begin
	gumbel_width = gumbel_width_from_fwhm(gumbel_fwhm)
	pulse_model = GumbelPulse(gumbel_width, pulse_amplitude)
	pulse = make_pulse_dist(pulse_model)
	gumbel_mode = evaluate_pulse_template(pulse_model, 0., [0.])[1]
end

# ╔═╡ eade10ef-2250-4297-81c4-13584067fb19
begin
	xs = -50:0.1:50
	plot(xs, evaluate_pulse_template(pulse_model, 0., xs),  ylabel="Amplitude (a.u.)", xlabel="Time (ns)")
	
end

# ╔═╡ 76105545-fa40-4029-ac40-fc4bf8bd4115
md"""
## Create an example waveform
Sampling frequency (GHz): $(@bind sampling_freq Slider(1:0.2:3, default=2, show_value=true)) 

SNRdb: $(@bind snr_db Slider(-10:1:30, default=10, show_value=true))
"""

# ╔═╡ 48ee34d3-056f-4ec7-b05e-f6d0c99bfcb6
begin
	snr = 10^(snr_db/10)
	noise_amp = gumbel_mode / snr
end

# ╔═╡ 46325f84-9f74-4f85-a6c0-cdbc5ce75bc2
begin
	Random.seed!(1337)
	arrival_time_pdf = Gamma(2., 5.)
	hit_times = rand(arrival_time_pdf, 50)
	
	charges = rand(spe_d, size(hit_times))
	
	#wf_model = make_gumbel_waveform_model(hit_times)
	wf = PulseSeries(hit_times, charges, pulse_model)
end


# ╔═╡ f4ac8ec0-128a-496f-9c0a-f8a3c89dd8c4
begin
	min_time, max_time = extrema(hit_times)
	min_time -= 30
	max_time += 30
	
	dt = 1/sampling_freq # ns
	timestamps_hires = range(min_time, max_time, step=dt)
	
	waveform_values = evaluate_pulse_series(timestamps_hires, wf)
	if noise_amp > 0
		waveform_values_noise = add_gaussian_white_noise(waveform_values, noise_amp)
	else
		waveform_values_noise = waveform_values
	end
	
	
	l = @layout [a ; b]
	p1 = plot(timestamps_hires, waveform_values, label="Waveform")
	p1 = plot!(timestamps_hires, waveform_values_noise, label="Waveform + Noise")
	
	p2 = histogram(hit_times, weights=charges, bins=0:1:100, label="Photons")
	plot(p1, p2, layout = l)
	
	
end

# ╔═╡ 795b4776-b631-4970-9b4c-0d0dc8d6c034
md"""
## Digitize waveform
ADC frequency (GHz): $(@bind adc_freq Slider(100E-3:10E-3:500E-3, default=200E-3, show_value=true))

Lowpass cutoff (GHz): $(@bind lp_cutoff Slider(50E-3:10E-3:500E-3, default=125E-3, show_value=true))


"""

# ╔═╡ 123b74c4-ade3-4f2a-acf7-953da84dd4dc
begin
	designmethod = Butterworth(1)
	filter = digitalfilter(Lowpass(lp_cutoff, fs=sampling_freq), designmethod)
end

# ╔═╡ d454fc4d-1c19-46e6-9864-3a61db3cdc5d
function plot_digitized_wf(
	timestamps_hires,
	timestamps,
	orig_wf,
	digi_wf_values)

	wf_eval = evaluate_waveform(timestamps_hires, orig_wf)
	
	plot(timestamps_hires, wf_eval, label="Waveform + Noise", 			xlabel="Time (ns)", ylabel="Amplitude (a.u.)", right_margin = 40Plots.px,
		xlim=(-20, 50))
	plot!(timestamps, digi_wf_values, label="Digitized Waveform")
	sticks!(twinx(), orig_wf.photon_times, orig_wf.photon_charges, legend=false, left_margin = 30Plots.px,
		ylabel="Charge (PE)", ylim=(1, 20), color=:green, xticks=:none)
end

# ╔═╡ f3916664-b222-46d7-a1dc-f3d40724d4ed
begin
	pulse_model_filt = make_filtered_pulse(pulse_model, sampling_freq, (-1000., 1000.), filter)
	

	single_pulse_wf = PulseSeries([0.], [1.], pulse_model)
	single_pulse_wf_eval = evaluate_pulse_series(timestamps_hires, single_pulse_wf)
	
	single_pulse_wf_filt = PulseSeries([0.], [1.], pulse_model_filt)
	single_pulse_wf_filt_eval = evaluate_pulse_series(timestamps_hires, single_pulse_wf_filt)
	
	plot(timestamps_hires, single_pulse_wf_eval, label="Original")
	plot!(timestamps_hires, single_pulse_wf_filt_eval, label="Filtered")
end

# ╔═╡ 22052e81-7862-4a07-81aa-e7dc9de467f8
begin
	#designmethod = Butterworth(1)
	#filter = digitalfilter(Lowpass(lp_cutoff, fs=sampling_freq), designmethod)
	
	digi_wf = digitize_waveform(
	    wf,
	    sampling_freq,
	    adc_freq,
	    noise_amp,
	    filter,
	    (min_time, max_time)
	)
	
	plot(timestamps_hires, waveform_values_noise, label="Waveform + Noise", 			xlabel="Time (ns)", ylabel="Amplitude (a.u.)", right_margin = 40Plots.px,
		xlim=(-20, 50))
	plot!(digi_wf.timestamps, digi_wf.values, label="Digitized Waveform")
	sticks!(twinx(), hit_times, charges, legend=false, left_margin = 30Plots.px, ylabel="Charge (PE)", ylim=(1, 20), color=:green, xticks=:none, xlim=(-20, 50))
end


# ╔═╡ 85d9b866-2ff8-4737-986f-072355ab18bd
md"""
## Unfolding

Initial pulse resolution (ns): $(@bind unf_pulse_res Slider(0.05:0.05:3, default=0.1, show_value=true))

"""

# ╔═╡ 20c8689f-7e86-4eea-887f-b74057513eeb
begin
	#pulse_times = collect(range(min_time, max_time, step=unf_pulse_res))
	#pulse_charges = apply_nnls(pulse_times, pulse_model_filt, digi_wf)

	pulse_times, pulse_charges = unfold_waveform(digi_wf, pulse_model_filt, unf_pulse_res, 0.2)

	orig_waveform = Waveform(collect(timestamps_hires), waveform_values_noise)
	
	plot_waveform(orig_waveform, digi_wf, pulse_times, pulse_charges, pulse_model, pulse_model_filt, (0., maximum(orig_waveform.values)*1.1))
end

# ╔═╡ b11aed7d-87c0-4e8c-b33e-e6013a56ee01
orig_waveform

# ╔═╡ abe4861a-98b3-463d-9234-fefd64cc3bff
pulse_times

# ╔═╡ 43fa1d20-8344-44e0-a3c3-038693d68831
pulse_charges

# ╔═╡ 80154f86-0fc2-46e5-bd3a-a64f5443476d
begin
	plot(ecdf(pulse_times, weights=pulse_charges), label="Pulse CDF")
	plot!(ecdf(hit_times), label="True CDF")
end

# ╔═╡ 6793e951-d0c2-4d8d-95dd-f8bb916e4381
md"""
## Resolutions
"""

# ╔═╡ 867c3a84-a6b1-4424-8246-8db3c767f6b0
md"""
### Single photon resolutions
"""


# ╔═╡ 9678bf67-145e-4344-b34f-c9261ee83c3b
function calc_spe_timing_resolution(
	n_samples::Integer,
	spe_d::ContinuousUnivariateDistribution,
	pulse_model::PulseTemplate{T},
	pulse_model_filt::PulseTemplate{T},
	pulse_resolution::T,
	min_charge::T) where {T <: Real}

	times::Vector{T} = rand(Uniform(-10., 10.), n_samples)
	charges::Vector{T} = rand(spe_d, n_samples)
	

	dt = []
	for (i, (t, c)) in enumerate(zip(times, charges))

		single_pulse_wf = PulseSeries([t], [c], pulse_model)
		
		digi_wf = digitize_waveform(
		    single_pulse_wf,
		    sampling_freq,
		    adc_freq,
		    noise_amp,
		    filter,
		    (min_time, max_time)
		)

		pulse_times, pulse_charges = unfold_waveform(digi_wf, pulse_model_filt, pulse_resolution, min_charge)

		if size(pulse_times, 1) > 0
			push!(dt, minimum(pulse_times) - t)
		end
	end
	dt
end

# ╔═╡ c4c1f925-973b-4525-a791-b5c4cabcd8d0
begin	
	dt_dist = calc_spe_timing_resolution(1000, spe_d, pulse_model, pulse_model_filt, unf_pulse_res, 0.25)
	dt_std = std(dt_dist)
	dt_iqr = iqr(dt_dist)
	h = histogram(dt_dist, label="STD: $(@sprintf("%.2f", dt_std)), IQR: $(@sprintf("%.2f", dt_iqr))")
	print(iqr(dt_dist))
	plot(h, xlabel="Δt (ns)", ylabel="Counts", label="")
end


# ╔═╡ 434eefb4-2b98-4db9-a004-f84f9e31fb78
md"""
### Multi-SPE Resolution
"""

# ╔═╡ 54fa9ff5-6849-4729-ae06-76cfcb2c1489
function plot_two_spe_pulse(
	pulse_distance::T,
	charge_ratio::T,
	spe_d::ContinuousUnivariateDistribution,
	pulse_model::PulseTemplate{T},
	pulse_model_filt::PulseTemplate{T},
	pulse_resolution::T,
	min_charge::T) where T

	charges = [1. ,charge_ratio]
	two_pulse_wf = PulseSeries([0., pulse_distance], charges, pulse_model)
				
		digi_wf_two_pulse = digitize_waveform(
			two_pulse_wf,
			sampling_freq,
			adc_freq,
			noise_amp,
			filter,
			(min_time, max_time)
		)
		
		pulse_times, pulse_charges = unfold_waveform(digi_wf_two_pulse, pulse_model_filt, pulse_resolution, min_charge)

		ev_times = collect(-50.:0.1:50.)
		ev_wf = evaluate_pulse_series(ev_times, two_pulse_wf)

		p = plot_waveform(Waveform(ev_times, ev_wf), digi_wf_two_pulse, pulse_times, pulse_charges, pulse_model, pulse_model_filt, (0., 25.))
end

	

# ╔═╡ 7772b118-2bd9-4ff9-8e7b-9d492c0eae40
function two_spe_unfolding(
	n_samples::Integer,
	pulse_distance::T,
	spe_d::ContinuousUnivariateDistribution,
	pulse_model::PulseTemplate{T},
	pulse_model_filt::PulseTemplate{T},
	pulse_resolution::T,
	min_charge::T) where {T <: Real}


	times::Vector{T} = rand(Uniform(-10., 10.), n_samples)
	charges::Matrix{T} = rand(spe_d, (n_samples, 2))
	
	dt = []
	p = plot()
	for (i, (t, c)) in enumerate(zip(times, eachrow(charges)))

		two_pulse_wf = PulseSeries([t, t+pulse_distance], c, pulse_model)
				
		digi_wf_two_pulse = digitize_waveform(
			two_pulse_wf,
			sampling_freq,
			adc_freq,
			noise_amp,
			filter,
			(min_time, max_time)
		)
		
		pulse_times, pulse_charges = unfold_waveform(digi_wf_two_pulse, pulse_model_filt, pulse_resolution, min_charge)

		ev_times = collect(-50.:0.1:50.)
		ev_wf = evaluate_pulse_series(ev_times, two_pulse_wf)
		
		println(pulse_times)
		println(pulse_charges)
		p = plot_waveform(Waveform(ev_times, ev_wf), digi_wf_two_pulse, pulse_times, pulse_charges, pulse_model, pulse_model_filt, (0., maximum(ev_wf)*1.1))
	end
	p
end

		

# ╔═╡ a411fb4b-46ff-4e21-90b9-5d3cc24edc22
md"""
Pulse separation(ns): $(@bind pulse_sep Slider(1.:1:20, default=5., show_value=true))

Charge ratio: $(@bind charge_ratio Slider(0.1:0.1:10, default=1., show_value=true))
"""

# ╔═╡ d62c174b-50b7-4676-bb60-5b99a99fd9cd

plot_two_spe_pulse(pulse_sep, charge_ratio, spe_d, pulse_model, pulse_model_filt, unf_pulse_res, 0.1, )

# ╔═╡ 0d464ee8-c0f2-489e-8045-87e16265bcd3
begin
	anim = @animate for ps in .5:0.1:15
		plot_two_spe_pulse(ps, charge_ratio, spe_d, pulse_model, pulse_model_filt, unf_pulse_res, 0.1, )
		title!("Separation: $(ps) (ns)")
	end
	gif(anim, "../figures/pulse_separation.gif", fps = 5)
	
end

# ╔═╡ 2aa2b928-e6e1-44fb-b0b4-d614fd3c4765
md"""
### Charge Resolution
"""

# ╔═╡ ad1d7e4c-a59b-465a-a1eb-f21910f1409e
begin
	function calc_charge_resolution(n_photons::Integer, n_samples::Integer, min_charge::T) where T
	
		tot_charges = []
		for i in 1:n_samples
			hit_times = rand(arrival_time_pdf, n_photons)
			charges = rand(spe_d, size(hit_times))
			wf = PulseSeries(hit_times, charges, pulse_model)
	
			digi_wf = digitize_waveform(
				wf,
				sampling_freq,
				adc_freq,
				noise_amp,
				filter,
				(min_time, max_time)
			)
			
			pulse_times, pulse_charges = unfold_waveform(digi_wf, pulse_model_filt, unf_pulse_res, min_charge)
	
			push!(tot_charges, sum(pulse_charges))
		end
		tot_charges
	end


	lognphs = 0:0.5:3

	nphs = floor.(10 .^lognphs)
	# label="μ:$(@sprintf("%.2f", mean(tot_charges))), σ: $(@sprintf("%.2f", std(tot_charges)))
	
	means = []
	stds = []
	
	for nph in nphs
		tot_charges = calc_charge_resolution(Int64(nph), 100, 0.1)
		push!(means, mean(tot_charges))
		push!(stds, std(tot_charges))
	end
	plot(nphs, means, yerr=stds, xscale=:log10, xlim=(1E-1, 1.5E3), yscale=:log10,
xlabel="Number of Photons", ylabel="Total Charge", label="")
	
end

# ╔═╡ d80535cf-fbec-445d-92e8-aa25b588ddc6
plot(nphs, stds, xscale=:log10, yscale=:log10)

# ╔═╡ 704edab1-2c4e-40d1-963f-e273fb2271b7
md"""
## Linearity Test
"""

# ╔═╡ a3cea64c-6abf-41f9-9882-6f7776aaecca
begin
	hit_times_1 = rand(arrival_time_pdf, 50)
	charges_1 = rand(spe_d, size(hit_times))
	wf_1 = PulseSeries(hit_times_1, charges_1, pulse_model)
	
	
	hit_times_2 = rand(arrival_time_pdf, 40) .+ 30.
	charges_2 = rand(spe_d, size(hit_times_1))
	wf_2 = PulseSeries(hit_times_2, charges_2, pulse_model)
	
	wf_combined = PulseSeries([hit_times_1 ; hit_times_2], [charges_1 ; charges_2], pulse_model)
	

	mint = -100.
	maxt = 100.
	
	digi_1 = digitize_waveform(wf_1, sampling_freq, adc_freq, noise_amp, filter, (mint, maxt))
	
	digi_2 = digitize_waveform(wf_2, sampling_freq, adc_freq, noise_amp, filter, (mint, maxt))
	
	digi_comb = digitize_waveform(wf_combined, sampling_freq, adc_freq, noise_amp, filter, (mint, maxt))

	#plot(wf_1, xlim=(-50, 50), label="True 1")
	#plot!(wf_2, label="True 2")
	plot(wf_combined, xlim=(-50, 100), label="True Combined", legend=:topleft)

	pulse_times_1, pulse_charges_1 = unfold_waveform(digi_1, pulse_model_filt, unf_pulse_res, 0.1)

	pulse_times_2, pulse_charges_2 = unfold_waveform(digi_2, pulse_model_filt, unf_pulse_res, 0.1)

	pulse_times_comb, pulse_charges_comb = unfold_waveform(digi_comb, pulse_model_filt, unf_pulse_res, 0.1)

	reco_wf_uf_1 = PulseSeries(pulse_times_1, pulse_charges_1, pulse_model)
	reco_wf_uf_2 = PulseSeries(pulse_times_2, pulse_charges_2, pulse_model)
	reco_wf_uf_comb = PulseSeries(pulse_times_comb, pulse_charges_comb, pulse_model)

	reco_wf_uf_comb_sum = reco_wf_uf_1 + reco_wf_uf_2
	
	#plot!(reco_wf_uf_1, label="Unfolded 1")
	#plot!(reco_wf_uf_2, label="Unfolded 2")
	plot!(reco_wf_uf_comb, label="Unfolded Combined")
	plot!(reco_wf_uf_comb_sum, label="Unfolded Summed")
	

	
end

# ╔═╡ 9085efee-5c24-4d6c-b00d-d461511310f3
begin
	a = [1, 2]
	b = [3, 4]
	c = [a; b]
end

# ╔═╡ 06f302ca-ce4c-40ba-8de8-47db908ed99b
md"""
## Unfolding the hit PDF
PDF -> Waveform -> Transformer NN -> Norm flow -> CRPS Score
"""

# ╔═╡ Cell order:
# ╠═c8f490da-d689-4f2b-8dcb-497c05973157
# ╟─ffd29632-7e6e-4925-b7e2-3814ee23ebb6
# ╠═3d7f7651-a64d-416c-9ea9-7455520fa7d7
# ╠═c45eec73-54a0-463c-a752-2e7bccb82310
# ╠═15f8b0e3-247b-4e91-b5d9-a01ef7a5e29b
# ╟─1adb8eb9-dd29-4eb5-af04-9d62b044b9ba
# ╠═f20109ca-878e-4407-88b2-21d31ea46ffb
# ╠═59635b6c-31ad-4749-bf71-8e55254fdbe3
# ╠═eade10ef-2250-4297-81c4-13584067fb19
# ╠═76105545-fa40-4029-ac40-fc4bf8bd4115
# ╠═48ee34d3-056f-4ec7-b05e-f6d0c99bfcb6
# ╠═46325f84-9f74-4f85-a6c0-cdbc5ce75bc2
# ╠═f4ac8ec0-128a-496f-9c0a-f8a3c89dd8c4
# ╠═795b4776-b631-4970-9b4c-0d0dc8d6c034
# ╠═123b74c4-ade3-4f2a-acf7-953da84dd4dc
# ╟─d454fc4d-1c19-46e6-9864-3a61db3cdc5d
# ╠═f3916664-b222-46d7-a1dc-f3d40724d4ed
# ╠═22052e81-7862-4a07-81aa-e7dc9de467f8
# ╠═85d9b866-2ff8-4737-986f-072355ab18bd
# ╠═20c8689f-7e86-4eea-887f-b74057513eeb
# ╠═b11aed7d-87c0-4e8c-b33e-e6013a56ee01
# ╠═abe4861a-98b3-463d-9234-fefd64cc3bff
# ╠═43fa1d20-8344-44e0-a3c3-038693d68831
# ╠═80154f86-0fc2-46e5-bd3a-a64f5443476d
# ╟─6793e951-d0c2-4d8d-95dd-f8bb916e4381
# ╟─867c3a84-a6b1-4424-8246-8db3c767f6b0
# ╠═9678bf67-145e-4344-b34f-c9261ee83c3b
# ╠═c4c1f925-973b-4525-a791-b5c4cabcd8d0
# ╠═434eefb4-2b98-4db9-a004-f84f9e31fb78
# ╠═54fa9ff5-6849-4729-ae06-76cfcb2c1489
# ╠═7772b118-2bd9-4ff9-8e7b-9d492c0eae40
# ╠═a411fb4b-46ff-4e21-90b9-5d3cc24edc22
# ╠═d62c174b-50b7-4676-bb60-5b99a99fd9cd
# ╠═0d464ee8-c0f2-489e-8045-87e16265bcd3
# ╠═2aa2b928-e6e1-44fb-b0b4-d614fd3c4765
# ╟─ad1d7e4c-a59b-465a-a1eb-f21910f1409e
# ╠═d80535cf-fbec-445d-92e8-aa25b588ddc6
# ╠═704edab1-2c4e-40d1-963f-e273fb2271b7
# ╠═a3cea64c-6abf-41f9-9882-6f7776aaecca
# ╠═9085efee-5c24-4d6c-b00d-d461511310f3
# ╠═06f302ca-ce4c-40ba-8de8-47db908ed99b
