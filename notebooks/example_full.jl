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

# ╔═╡ 3e649cbf-1546-4204-82de-f6db5be401c7
begin
    import Pkg
    Pkg.activate("..")

	using PlutoUI
	using Plots
	using StatsPlots
	using Parquet
	using StaticArrays
	using Unitful
	using LinearAlgebra
	using Distributions
	using Base.Iterators
	using Hyperopt
	using Random
	using Interpolations
	using StatsFuns
	using LogExpFunctions
	using DataFrames
	using NeutrinoTelescopes.PhotonPropagationCuda
	using NeutrinoTelescopes.Medium
	using NeutrinoTelescopes.Detection
	using NeutrinoTelescopes.LightYield
	using NeutrinoTelescopes.Modelling
	using NeutrinoTelescopes.Spectral
	
	using Flux
	using BSON: @save, @load
	using BSON
end

# ╔═╡ db772593-2e58-4cec-bc88-7113ac028811
using Zygote

# ╔═╡ 5ab60c75-fd19-4456-9121-fb42ce3e086f
md"""
# Photon detection model for EM cascade segments
"""

# ╔═╡ a615b299-3868-4083-9db3-806d7390eca1
md"""
## Create a medium and propagate photons
Propagate photons from an isotropic Cherenkov emitter to a spherical detector.

Distance: $(@bind distance Slider(5f0:0.1f0:200f0, default=25, show_value=true)) m

"""


# ╔═╡ 4db5a2bd-97bc-412c-a1d3-cb8391425e20
begin
	n_photons = 100000
	medium = Medium.make_cascadia_medium_properties(Float32)
	prop_result = PhotonPropagationCuda.propagate_distance(distance, medium, n_photons)
	nothing
end

# ╔═╡ b37de33c-af67-4983-9932-2c32e8db399f
@df prop_result corrplot(cols(1:5))

# ╔═╡ 911c5c61-bcaa-4ba4-b5c0-09a112b6e877
md"""
The resulting DataFrame contains information about the detected photons:
- time-residual (`tres`)
- initial emission angle (`initial_theta`)
- refractive index for the photon's wavelength (`ref_ix`) 
- absorption weight (`abs_weight`)

The time-residual is the photon arrival time relative to the geometric path length 
```math
\begin{align}
t_{res} = t - t_{geo}(800 \,\mathrm{nm}) \\
t_{geo}(\lambda) = \frac{\mathrm{distance}}{\frac{c_{vac}}{n(\lambda)}}
\end{align}
```

Note: The detector is assumed to be 100% efficient. Lower detection efficiences can be trivialled added by an additional (wl-dependent) weight.

Photons can be reweighted to a Cherenkov angular emission spectrum:

"""




# ╔═╡ 30c4a297-715d-4268-b91f-22d0cac66511
begin
	my_obs_angle = deg2rad(70)
	dir_reweight = Modelling.get_dir_reweight(
		prop_result[:, :initial_theta],
		my_obs_angle,
		prop_result[:, :ref_ix])
	total_weight = dir_reweight .* prop_result[:, :abs_weight]
	histogram(prop_result[:, :tres], weights=total_weight, label="", xlabel="Time Residual (ns)", ylabel="Detected Photons")
end

# ╔═╡ 26469e97-51a8-4c00-a69e-fe50ad3a625a
md"""
## Fit Distributions
Runs the photon propagation for multiple distances and reweights those simulations to multiple observation angles. Fits the resulting arrival time distributions with Gamma-PDFs.
"""

# ╔═╡ 74b11928-fe9c-11ec-1d37-01f9b1e48fbe
begin	
	#results_df = Modelling.make_photon_fits(Int64(1E8), 250, 250, 300f0)
	#write_parquet("photon_fits.parquet", results_df)
	results_df = read_parquet("../assets/photon_fits.parquet")
end#


# ╔═╡ 26c7461c-5475-4aa9-b87e-7a55b82cea1f
@df results_df corrplot(cols(1:5))

# ╔═╡ 78695b21-e992-4f5c-8f34-28c3027e3179
@df results_df scatter(:distance, :det_fraction, yscale=:log10)

# ╔═╡ b9af02cb-b675-4030-b3c8-be866e85ebc7
md"""
## Fit distribution parameters with MLP
Here we fit a simple MLP to predict the distribution parameters (and the photon survival rate) as function of the distance and the observation angle
"""

# ╔═╡ cb97110d-97b2-410d-8ce4-bef9accfcef2
begin

	#=
	params = Dict(
		:width=>1024,
		:learning_rate=>0.001,
		:batch_size=>4096,
		:data_file=>"../assets/photon_fits.parquet",
		:dropout_rate=>0.1,
		:seed=>31138,
		:epochs=>250
		)
	
	
	# Parameters from hyperparam optimization

	
	model, train_data, test_data, trafos = Modelling.train_mlp(;params...)
	@show Modelling.loss_all(test_data, model)

	model = cpu(model)
	@save "photon_model.bson" model params

	=#
	model = BSON.load("../assets/photon_model.bson", @__MODULE__)[:model]
	params = BSON.load("../assets/photon_model.bson", @__MODULE__)[:params]
	#@load "photon_model.bson" model

	
	model = gpu(model)
	train_data, test_data, trafos = Modelling.get_data(Modelling.Hyperparams(;params...))

	train_data = gpu.(train_data)
	test_data = gpu.(test_data)
		
	output_trafos = [
		trafos[(:fit_alpha, :log_fit_alpha)],
		trafos[(:fit_theta, :log_fit_theta)],
		trafos[(:det_fraction, :log_det_fraction_scaled)]
	]
	

end

# ╔═╡ 8638dde4-9f02-4dbf-9afb-32982390a0b6
begin
	feat_test = hcat(collect(td[1] for td in test_data)...) 
	targ_test = hcat(collect(td[2] for td in test_data)...) |> cpu
	target_pred_test = model(feat_test) |> cpu
	feat_test = cpu(feat_test)

	feat_train = hcat(collect(td[1] for td in train_data)...) 
	targ_train = hcat(collect(td[2] for td in train_data)...) |> cpu
	target_pred_train = model(feat_train) |> cpu
	feat_train = cpu(feat_train)
	
	
	l = @layout [a b c; d e f]
	plots = []
	feature_names = [:log_distance, :cos_obs_angle]
	target_names = [:log_fit_alpha, :log_fit_theta, :log_det_fraction]
	
	for (i, j) in Base.product(1:3, 1:2)
	    p = scatter(feat_test[j, :], targ_test[i, :], alpha=0.7, label="Truth",
	        xlabel=feature_names[j], ylabel=target_names[i], ylim=(-1, 2), legend=:topleft)
	    scatter!(p, feat_test[j, :], target_pred_test[i, :], alpha=0.7, label="Prediction")
	    push!(plots, p)
	end

	plot(plots..., layout=l)
	
end

# ╔═╡ 984b11c9-2cd5-4dd6-9f36-26eec69eb17d
begin
	histogram(targ_train[3, :] - target_pred_train[3, :], normalize=true)
	histogram!(targ_test[3, :] - target_pred_test[3, :], normalize=true)
	
	
end

# ╔═╡ 0a37ce94-a949-4c3d-9cd7-1a64b1a3ce47
md"""
## Compare model prediction to MC
"""

# ╔═╡ be1a6f87-65e3-4741-8b98-bb4d305bd8c3
begin
    position = @SVector[0.0, 0.0, 0.0]
    direction = @SVector[0.0, 0.0, 1.0]
    energy = 1E5
    time = 0.0
	photons = 1000000
	medium64 = Medium.make_cascadia_medium_properties(Float64)
	spectrum = Spectral.CherenkovSpectrum((300.0, 800.0), 20, medium64)
	em_profile = Emission.AngularEmissionProfile{:IsotropicEmission, Float64}()
	
	source = LightYield.CherenkovSegment(position, direction, time, Float64(photons))
	

	target_pos = @SVector[-10., 0.0, 10.0]
	target = Detection.DetectionSphere(target_pos, 0.21)
	
	model_input = Matrix{Float32}(undef, (2, 1))
	model_input[:, 1] .= Modelling.source_to_input(source, target)
	model_pred = cpu(model(gpu(model_input)))

	Modelling.transform_model_output!(model_pred, output_trafos)
	
	n_photons_pred = photons * model_pred[3, 1]

	
	this_prop_result = PhotonPropagation.propagate_distance(Float32(exp10(model_input[1, 1])), medium, photons)

	this_total_weight = (
		Modelling.get_dir_reweight(this_prop_result[:, :initial_theta],
			acos(model_input[2, 1]), this_prop_result[:, :ref_ix])
		.* this_prop_result[:, :abs_weight]
		.* Detection.p_one_pmt_acc.(this_prop_result[:, :wavelength])
	)
	
	histogram(this_prop_result[:, :tres], weights=this_total_weight, normalize=false,
	xlabel="Time (ns)", ylabel="# Photons", label="MC")

	xs_plot = 0:0.1:15
	
	gamma_pdf = Gamma(model_pred[1], model_pred[2])
	
	plot!(xs_plot, n_photons_pred .* pdf.(gamma_pdf, xs_plot), label="Model")
	#n_photons_pred, sum(this_total_weight)
end





# ╔═╡ 57450082-04dc-4d1a-8ef4-e321c3971c84
n_photons_pred, sum(this_total_weight)

# ╔═╡ 31ab7848-b1d8-4380-872e-8a12c3331051
md"""
## Use model to simulate an event for a detector.

The "particle" (EM-cascade) is split into a series of point-like Cherenov emitters, aligned along the cascade axis. Their respective lightyield is calculated from the longitudinal profile of EM cascades.
"""

# ╔═╡ 22e419a4-a4f9-48d5-950b-8420854c475a
begin
	Emission.frank_tamm_norm((300., 800.), wl -> Medium.get_refractive_index(wl, medium)) * LightYield.cherenkov_track_length(1E5, LightYield.EMinus)	
	
end

# ╔═╡ 4f7a9060-33dc-4bb8-9e41-eae5b9a30aa6


# ╔═╡ 40940de0-c45b-4236-82f0-54a77d5fbb9a
begin
	positions = Detection.make_detector_cube(5, 5, 10, 50., 100.)
	
	xs = [p[1] for p in positions]
	ys = [p[2] for p in positions]
	
	scatter(xs, ys)
end

# ╔═╡ 709a7e6f-1e54-4350-b826-66b9097cc46a


# ╔═╡ 820e698c-f1c4-4d07-90d8-a4a13ff9cccd
md"""
### Calculate likelihoods

Particle zenith: $(@bind zenith_angle Slider(0:0.1:180, default=25, show_value=true)) deg

Particle azimuth $(@bind azimuth_angle Slider(0:0.1:360, default=180, show_value=true)) deg

"""

# ╔═╡ 462567a7-373f-4ac6-a11a-4c6853a8c45a

begin

	
	
	p1 = plot(poissons[136])
	xplot = 0.01:0.1:20
	p2  = plot(xplot, loglikelihood.(shapes[136], xplot), ylim=(-10, 0))
	plot(p1, p2)
end

# ╔═╡ 0cebefc7-0397-4f53-b423-f5a94bde3180


# ╔═╡ ec3dbcbe-028a-4700-a711-bdac55255494
begin

	struct LossFunction{T}
		x::T
		y::T
		z::T
		theta::T
		phi::T
		log_energy::T		
	end


	function eval_loggamma_mix(logweights, αs, θs, x)
		gamma_evals = gammalogpdf.(αs, θs, x)
		LogExpFunctions.logsumexp(logweights .+ gamma_evals)	
	end

	
	
	function (l::LossFunction{T})(
		target::Detection.PhotonTarget{T},
		times::Vector{T},
		int_grid::AbstractVector{T}) where {T<:Real}

		targets = [target]
		nph = [length(times)]
		event = [times]

		n_targets = 1
		n_sources = length(int_grid)-1
		
		position = SA[l.x, l.y, l.z]
		direction =  sph_to_cart(l.theta, l.phi)
		time = 0.0

		particle = LightYield.Particle(
			position,
			direction,
			time,
			exp10(l.log_energy),
			LightYield.EMinus
		)

		source_out = Vector{LightYield.CherenkovSegment{T}}(undef, n_sources)

		source_out_buf = Zygote.Buffer(source_out)
		LightYield.particle_to_elongated_lightsource!(
			particle,
			int_grid,
			medium,
			(300.0, 800.0),
			source_out_buf)
		source_out = copy(source_out_buf)

		
		inputs = Matrix{Float32}(undef, (2, n_targets * n_sources))
		inp_buf = Zygote.Buffer(inputs)

		for (i, (source, target)) in enumerate(product(source_out, targets))
			res = Modelling.source_to_input(source, target)
			inp_buf[1, i] = res[1]
			inp_buf[2, i] = res[2] 
		end

		inputs = copy(inp_buf)
		predictions = cpu(model(gpu(inputs)))
		predictions = Modelling.reverse_transformation.(predictions, output_trafos)
		predictions = reshape(predictions, (3, n_sources, n_targets))
	
		poissons = poisson_dist_per_module(predictions, source_out)
		shapes = shape_mixture_per_module(predictions)

		mask = nph == 0

		shape_lh = zeros(n_targets)
		shape_lh_buf = Zygote.Buffer(shape_lh)


		for i in 1:n_targets
			if nph[i] > 0
				shape_lh_buf[i] = loglikelihood(shapes[i], times[i])				
			else
				shape_lh_buf[i] = 0.
			end
		end
		shape_lh = copy(shape_lh_buf)
		
		sum(shape_lh .+ loglikelihood.(poissons, nph))

		#=
		lh_sum = 0
		for j in 1:n_targets

			if nph[j] > 0
				log_weighting = log.(predictions[3, :, j] ./ sum(predictions[3, :, j]))

				lh_sum += sum(eval_loggamma_mix.((log_weighting, ), (predictions[1, :, j], ), (predictions[2, :, j], ), event[j]))
			end
		
			nph_expec = [sum([predictions[3, i, j] * source_out[i].photons for i in 1:n_sources]) for j in 1:n_targets]
			lh_sum += sum(poislogpdf.(nph_expec, nph))
		end
		lh_sum
		=#
	end

	function eval_fisher(particle, targets, event, precision)
	
		len_range = (0., 20.)
			
		int_grid = range(len_range[1], len_range[2], step=precision)
		n_steps = size(int_grid, 1)
	
		ptheta = acos(particle.direction[3])
		pphi = acos(particle.direction[1] / sin(ptheta))
		lfunc = LossFunction(
			particle.position.x,
			particle.position.y,
			particle.position.z,
			ptheta,
			pphi,
			log10(particle.energy)
			)

		gradients = [Zygote.gradient(model -> model(target, times, int_grid), lfunc)[1] for (target, times) in zip(targets, event)]

		gradients_df = DataFrame(gradients)
		grad_vec = reshape(sum.(eachcol(gradients_df)), (1,6))
		fisher_info = grad_vec .* grad_vec'
			   
	end

end



# ╔═╡ 4851b215-49e4-4338-9dbe-815130fe6074
begin
	event = sample_event(poissons, shapes)
	eval_fisher(particle, targets, event, 0.5)
end

# ╔═╡ a78ce00a-a3cd-4d56-9747-d3b8afa6f1be
begin
	precisions = 0.1:0.1:2
	crs = []
	for prec in precisions
		poissons = poisson_dist_per_module(targets, particle, medium64, 0.5)
		shapes = shape_mixture_per_module(targets, particle, medium64, 0.5)
		events = [sample_event(poissons, shapes) for _ in 1:10]	
		fishers = map(ev -> eval_fisher(particle, targets, ev, prec), events)
		cr = sqrt.(diag(inv(sum(fishers))))
		push!(crs, cr)
	end
end

# ╔═╡ 5c843b58-1291-4f5d-99e9-e2eae49ac37f
plot(precisions, [cr[5] for cr in crs])

# ╔═╡ 9e1137fc-21e5-40ce-83a6-ddfbaa791fc2


# ╔═╡ 84457b21-8550-4510-84bd-d89caa29693f
begin
	gm = reshape([g for g in grad[1]], (4, 1))
	
	gm .* gm'
end

# ╔═╡ 664e4d6a-93ef-40f2-a585-b3a94aa94cea
begin
	function testf(xs::T) where T
		sum(map(x -> 2*x, xs))
	end
	
	Zygote.gradient(testf, [1., 2., 3.])
end

# ╔═╡ Cell order:
# ╠═3e649cbf-1546-4204-82de-f6db5be401c7
# ╟─5ab60c75-fd19-4456-9121-fb42ce3e086f
# ╠═a615b299-3868-4083-9db3-806d7390eca1
# ╠═4db5a2bd-97bc-412c-a1d3-cb8391425e20
# ╠═b37de33c-af67-4983-9932-2c32e8db399f
# ╠═911c5c61-bcaa-4ba4-b5c0-09a112b6e877
# ╠═30c4a297-715d-4268-b91f-22d0cac66511
# ╟─26469e97-51a8-4c00-a69e-fe50ad3a625a
# ╠═74b11928-fe9c-11ec-1d37-01f9b1e48fbe
# ╠═26c7461c-5475-4aa9-b87e-7a55b82cea1f
# ╠═78695b21-e992-4f5c-8f34-28c3027e3179
# ╠═b9af02cb-b675-4030-b3c8-be866e85ebc7
# ╠═cb97110d-97b2-410d-8ce4-bef9accfcef2
# ╠═8638dde4-9f02-4dbf-9afb-32982390a0b6
# ╠═984b11c9-2cd5-4dd6-9f36-26eec69eb17d
# ╠═0a37ce94-a949-4c3d-9cd7-1a64b1a3ce47
# ╠═be1a6f87-65e3-4741-8b98-bb4d305bd8c3
# ╠═57450082-04dc-4d1a-8ef4-e321c3971c84
# ╠═31ab7848-b1d8-4380-872e-8a12c3331051
# ╠═22e419a4-a4f9-48d5-950b-8420854c475a
# ╠═4f7a9060-33dc-4bb8-9e41-eae5b9a30aa6
# ╠═40940de0-c45b-4236-82f0-54a77d5fbb9a
# ╠═709a7e6f-1e54-4350-b826-66b9097cc46a
# ╠═820e698c-f1c4-4d07-90d8-a4a13ff9cccd
# ╠═462567a7-373f-4ac6-a11a-4c6853a8c45a
# ╠═db772593-2e58-4cec-bc88-7113ac028811
# ╠═0cebefc7-0397-4f53-b423-f5a94bde3180
# ╠═ec3dbcbe-028a-4700-a711-bdac55255494
# ╠═4851b215-49e4-4338-9dbe-815130fe6074
# ╠═a78ce00a-a3cd-4d56-9747-d3b8afa6f1be
# ╠═5c843b58-1291-4f5d-99e9-e2eae49ac37f
# ╠═9e1137fc-21e5-40ce-83a6-ddfbaa791fc2
# ╠═84457b21-8550-4510-84bd-d89caa29693f
# ╠═664e4d6a-93ef-40f2-a585-b3a94aa94cea
