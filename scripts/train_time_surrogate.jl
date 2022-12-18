using NeutrinoTelescopes
using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
using BenchmarkTools
using Random
using TensorBoardLogger

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


rng = MersenneTwister(31338)
nsel_frac = 0.3
tres, nhits, cond_labels, tf_dict = read_hdf(fnames, nsel_frac, rng)


nit = 100

hob = @hyperopt for i = nit,
    sampler = CLHSampler(dims=[Continuous(), Categorical(4), Continuous(), Categorical(11), Categorical(4)]),
    lr = 10 .^ LinRange(-4, -2, nit),
    mlp_layer_size = [256, 512, 768, 1024],
    dropout = LinRange(0, 0.5, nit),
    K = 5:15,
    batch_size = [512, 1024, 2048, 4096]

    model, model_loss, hparams, opt = train_time_expectation_model(
        (tres=tres, label=cond_labels, nhits=nhits),
        true,
        true,
        K=Int(K),
        epochs=100,
        lr=lr,
        mlp_layer_size=mlp_layer_size,
        mlp_layers=2,
        dropout=dropout,
        non_linearity=:relu,
        batch_size=batch_size,
        seed=1,)

    model_path = joinpath(@__DIR__, "../assets/rq_spline_model_K$(K)_LR$(lr)_MLP_$(mlp_layer_size)_DRP_$(dropout)_BS_$(batch_size).bson")
    @save model_path cpu(model) hparams opt tf_dict
    model_loss
end

hob

ho_path = joinpath(@__DIR__, "../assets/hyperopt.bson")
@save hob.history hob.results
