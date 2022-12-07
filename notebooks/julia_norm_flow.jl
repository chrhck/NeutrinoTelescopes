using NeutrinoTelescopes
using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
using BenchmarkTools
using Random
using StaticArrays

using OneHotArrays

using StatsBase
using Hyperopt
using Flux
using Plots
using BSON: @save, @load


fnames = [
    joinpath(@__DIR__, "../assets/photon_table_extended_2.hd5"),
    joinpath(@__DIR__, "../assets/photon_table_extended_3.hd5"),
    joinpath(@__DIR__, "../assets/photon_table_extended_4.hd5"),
    joinpath(@__DIR__, "../assets/photon_table_extended_5.hd5")
    ]
rng = MersenneTwister(31338)
nsel_frac = 0.9
tres, nhits, cond_labels, tf_dict = read_hdf(fnames, nsel_frac, rng)

model, model_loss, hparams, opt = train_time_expectation_model(
        (tres=tres, label=cond_labels, nhits=nhits),
        true,
        true,
        epochs=100,
        lr=0.001,
        mlp_layer_size=768,
        mlp_layers=2,
        dropout=0,
        non_linearity=:relu,
        batch_size=1000,
        seed=1)

model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")
model = cpu(model)
@save model_path model hparams opt tf_dict

model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")
@load model_path model hparams opt tf_dict


Flux.testmode!(model)

h5open(fnames[2], "r") do fid
    df = create_pmt_table(fid["pmt_hits"]["dataset_1500"])
    plot_tres = df[:, :tres]
    plot_labels, _ = preproc_labels(df, tf_dict)


    n_per_pmt = combine(groupby(df, :pmt_id), nrow => :n)
    max_i = argmax(n_per_pmt[:, :n])


    pmt_plot = n_per_pmt[max_i, :pmt_id]
    mask = df.pmt_id .== pmt_plot

    plot_tres_m = plot_tres[mask]
    plot_labels_m = plot_labels[mask, :]


    t_plot = -5:0.1:50
    l_plot = repeat(Vector(plot_labels[1, :]), 1, length(t_plot))


    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Shape", xlabel="Time Residual (800nm) (ns)", ylabel="PDF")
    ax2 = Axis(fig[1, 2], title="Shape + Counts", xlabel="Time Residual (800nm) (ns)", ylabel ="Counts")

    log_pdf_eval, log_expec = cpu(model)(t_plot, l_plot, true)

    lines!(ax1, t_plot, exp.(log_pdf_eval))
    hist!(ax1, plot_tres[mask], bins=-5:1:50, normalization=:pdf)

    lines!(ax2, t_plot, exp.(log_pdf_eval) .* exp.(log_expec))
    hist!(ax2, plot_tres[mask], bins=-5:1:50)

    println("$(exp(log_expec[1])), $(sum(mask))")

    fig
end


pos = SA[0., 20., 0.]
dir_theta = deg2rad(20)
dir_phi = deg2rad(50)
dir = sph_to_cart(dir_theta, dir_phi)

pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0

p = Particle(pos, dir, 0., 1E5, PEPlus)
target = MultiPMTDetector(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float32),
        UInt16(1)
    )


input = calc_flow_inputs([p], [target], traf)

get_pmt_count(typeof(target))

fig = Figure()
ax = Axis(fig[1, 1])
t_plot = -5:0.1:50
for i in 1:16
    l_plot = repeat(Vector(input[:, i]), 1, length(t_plot))
    lines!(ax, t_plot, exp.(model(t_plot, l_plot, true)[1]))
end
fig

#summary_data = ds_summary[:, [:nhits, :hittime_mean, :hittime_std]]


#tr_summary_data = fit_transform!(norm_targ, summary_data) |> Matrix |> adjoint



ho = @hyperopt for i = 100,
    sampler = RandomSampler(),
    batch_size = [50, 100, 200, 500],
    lr = 10 .^ (-3:0.5:-1.5),
    mlp_layer_size = [64, 128, 256, 512],
    dropout = [0, 0.1, 0.2]

    model, loss = train_expectation_model(
            (data=summary_data, label=tr_summary_labels),
            true,
            true,
            epochs=200,
            lr=lr,
            mlp_layer_size=mlp_layer_size,
            mlp_layers=2,
            dropout=dropout,
            non_linearity=:relu,
            batch_size=batch_size,
            seed=1)
    loss
end


Plots.plot(ho)

   







hist!(fig[1, 1], df[df.pmt_id.==pmt_plot, :tres], bins=-5:0.5:50, normalization=:pdf)

plot_labels

tr_cond_labels






l_plot = onehotbatch(fill(pmt_plot, length(t_plot)), 1:16)

lines(fig[1, 1], t_plot, exp.(rq_layer(t_plot, l_plot)))
hist!(fig[1, 1], df[df.pmt_id.==pmt_plot, :tres], bins=-5:0.5:50, normalization=:pdf)

fig


first(data)

rq_layer(randn(100), randn((16, 100)))


rq = Bijectors.RationalQuadraticSpline(widths, heights, derivs, B)
b = Bijectors.Scale(10) ∘ rq

td = transformed(base, b)
xs = -50:0.1:50
lines!(fig[1, 1], xs, pdf.(td, xs))
fig




Flux.params(widths, heights, derivs)
=#