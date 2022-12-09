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
rng = MersenneTwister(31338)
nsel_frac = 1
tres, nhits, cond_labels, tf_dict = read_hdf(fnames, nsel_frac, rng)

model, model_loss, hparams, opt = train_time_expectation_model(
        (tres=tres, label=cond_labels, nhits=nhits),
        true,
        true,
        K=12,
        epochs=100,
        lr=0.001,
        mlp_layer_size=768,
        mlp_layers=2,
        dropout=0.1,
        non_linearity=:relu,
        batch_size=1000,
        seed=1,)

model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")
model = cpu(model)
@save model_path model hparams opt tf_dict