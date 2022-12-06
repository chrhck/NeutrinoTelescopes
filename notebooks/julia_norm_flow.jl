using NeutrinoTelescopes
using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
using BenchmarkTools
using Random

using AutoMLPipeline

using StatsBase
using Hyperopt
using Flux
using Plots
using BSON: @save


fnames = [
    joinpath(@__DIR__, "../assets/photon_table_extended_2.hd5"),
    joinpath(@__DIR__, "../assets/photon_table_extended_3.hd5"),
    joinpath(@__DIR__, "../assets/photon_table_extended_4.hd5"),
    joinpath(@__DIR__, "../assets/photon_table_extended_5.hd5")
    ]
rng = MersenneTwister(31338)
nsel_frac = 0.9
tres, nhits, cond_labels = read_hdf(fnames, nsel_frac, rng)
tr_cond_labels, traf = fit_trafo_pipeline(cond_labels)


ho = @hyperopt for i = 50,
    sampler = RandomSampler(),
    batch_size = [1000, 2000, 5000, 10000],
    lr = 10 .^ (-3:0.5:-1.5),
    mlp_layer_size = [256, 512, 768, 1024],
    dropout = [0, 0.1, 0.2],
    non_linearity = [:relu, :tanh]


    model, loss = train_time_model(
        (tres=tres, label=tr_cond_labels, nhits=nhits),
        true,
        true,
        epochs=30,
        lr=lr,
        mlp_layer_size=mlp_layer_size,
        dropout=dropout,
        non_linearity=non_linearity,
        batch_size=batch_size,
    )
    loss
end

ho

ho_res = Plots.plot(ho, yscale=:log10)

Plots.savefig(ho_res, joinpath(@__DIR__, "../figures/hyperopt.png"))


model, model_loss, hparams, opt = train_time_expectation_model(
        (tres=tres, label=tr_cond_labels, nhits=nhits),
        true,
        true,
        epochs=100,
        lr=0.001,
        mlp_layer_size=512,
        mlp_layers=2,
        dropout=0,
        non_linearity=:relu,
        batch_size=1000,
        seed=1)

model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")
model = cpu(model)
@save model_path model hparams


Flux.testmode!(model)

h5open(fnames[2], "r") do fid
    df = create_pmt_table(fid["pmt_hits"]["dataset_2500"])
    plot_tres = df[:, :tres]
    plot_labels = preproc_labels(df)

    @show combine(groupby(plot_labels, :pmt_id), nrow => :n)

    n_per_pmt = combine(groupby(plot_labels, :pmt_id), nrow => :n)
    max_i = argmax(n_per_pmt[:, :n])


    pmt_plot = n_per_pmt[max_i, :pmt_id]
    mask = plot_labels.pmt_id .== pmt_plot

    plot_tres_m = plot_tres[mask]
    plot_labels_m = plot_labels[mask, :]

    plot_labels_m_tf = AutoMLPipeline.transform(traf, plot_labels_m)


    t_plot = -5:0.1:50
    l_plot = repeat(Vector(plot_labels_m_tf[1, :]), 1, length(t_plot))


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