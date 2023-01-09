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
    joinpath(@__DIR__, "../data/photon_table_extended_2.hd5"),
    joinpath(@__DIR__, "../data/photon_table_extended_3.hd5"),
    joinpath(@__DIR__, "../data/photon_table_extended_4.hd5"),
    joinpath(@__DIR__, "../data/photon_table_extended_5.hd5"),
    joinpath(@__DIR__, "../data/photon_table_extended_6.hd5")
]

model_name = "rq_spline_model_l2_0"

rng = MersenneTwister(31338)

for model_num in 1:5
    nsel_frac = 0.5
    tres, nhits, cond_labels, tf_dict = read_pmt_hits(fnames, nsel_frac, rng)

    chk_path = joinpath(@__DIR__, "../assets/$(model_name)_$model_num")

    model, model_loss, hparams, opt = train_time_expectation_model(
        (tres=tres, label=cond_labels, nhits=nhits),
        true,
        true,
        chk_path,
        K=12,
        epochs=100,
        lr=0.001,
        mlp_layer_size=768,
        mlp_layers=2,
        dropout=0.1,
        non_linearity=:relu,
        batch_size=10000,
        seed=1,
        l2_norm_alpha=0)

    model_path = joinpath(@__DIR__, "../assets/$(model_name)_$(model_num)_FNL.bson")
    model = cpu(model)

    @save model_path model hparams opt tf_dict
end

nsel_frac = 1
tres, nhits, cond_labels, tf_dict = read_pmt_hits(fnames, nsel_frac, rng)

chk_path = joinpath(@__DIR__, "../assets/$(model_name)_FULL")

model, model_loss, hparams, opt = train_time_expectation_model(
    (tres=tres, label=cond_labels, nhits=nhits),
    true,
    true,
    chk_path,
    K=12,
    epochs=100,
    lr=0.001,
    mlp_layer_size=768,
    mlp_layers=2,
    dropout=0.1,
    non_linearity=:relu,
    batch_size=10000,
    seed=1,
    l2_norm_alpha=0)

model_path = joinpath(@__DIR__, "../assets/$(model_name)_FULL_FNL.bson")
model = cpu(model)

@save model_path model hparams opt tf_dict




begin
    @load joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_5_FNL.bson") model hparams opt tf_dict
    @load joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_5_BEST.bson") model
end

h5open(fnames[2], "r") do fid
    df = create_pmt_table(fid["pmt_hits"]["dataset_4150"])
    plot_tres = df[:, :tres]
    plot_labels, _ = preproc_labels(df, 16, tf_dict)


    n_per_pmt = combine(groupby(df, :pmt_id), nrow => :n)
    max_i = argmax(n_per_pmt[:, :n])


    pmt_plot = n_per_pmt[max_i, :pmt_id]
    mask = df.pmt_id .== pmt_plot

    plot_tres_m = plot_tres[mask]
    plot_labels_m = plot_labels[:, mask]


    t_plot = -5:0.1:50
    l_plot = repeat(Vector(plot_labels[:, 1]), 1, length(t_plot))


    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Shape", xlabel="Time Residual (800nm) (ns)", ylabel="PDF")
    ax2 = Axis(fig[1, 2], title="Shape + Counts", xlabel="Time Residual (800nm) (ns)", ylabel="Counts")

    log_pdf_eval, log_expec = cpu(model)(t_plot, l_plot, true)

    lines!(ax1, t_plot, exp.(log_pdf_eval))
    hist!(ax1, plot_tres[mask], bins=-5:1:50, normalization=:pdf)

    lines!(ax2, t_plot, exp.(log_pdf_eval) .* exp.(log_expec))
    hist!(ax2, plot_tres[mask], bins=-5:1:50)

    println("$(exp(log_expec[1])), $(sum(mask))")

    fig
end
