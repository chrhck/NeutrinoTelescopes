using NeutrinoTelescopes
using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
using BenchmarkTools
using Random
using TensorBoardLogger
using AutoMLPipeline

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
nsel_frac = 0.1
tres, nhits, cond_labels = read_hdf(fnames, nsel_frac, rng)
tr_cond_labels, traf = fit_trafo_pipeline(cond_labels)

function run_hyperopt(data)
    inner_opt = BOHB(dims=[
        Hyperopt.Continuous(),
        Hyperopt.Continuous(),
        Hyperopt.Continuous(),
        Hyperopt.Continuous()]
        )


    local model_seed = 1
    
    bohb = @hyperopt for resources=50,
        sampler=Hyperband(R=50, η=3, inner=inner_opt),
        batch_size = [1000, 2000, 5000, 10000],
        lr = 10 .^ (-3:0.5:-1.5),
        mlp_layer_size = LinRange(128, 1024, 5),
        dropout = LinRange(0., 0.3, 20)

        epochs = Int(ceil(resources*5))


        if state !== nothing
            println("Continuing training with epochs=$epochs")
            model_path = joinpath(@__DIR__, "../assets/rq_spline_model_hopt_$(state[1]).bson")

            @load model_path model hparams opt

            train_loader, test_loader = setup_dataloaders(data, hparams)

            device = gpu
            logdir = joinpath(@__DIR__, "../../tensorboard_logs/RQNormFlowHyperopt")
            lg = TBLogger(logdir)
            model, model_loss = train_model!(opt, train_loader, test_loader, model, log_likelihood_with_poisson, hparams, lg, device, true)

            return model_loss, (hparams.seed, )
            
        else
            println("New model with epochs=$epochs")
            model, model_loss, hparams, opt = train_time_expectation_model(
                data,
                true,
                true,
                epochs=epochs,
                lr=lr,
                mlp_layer_size=Int(mlp_layer_size),
                mlp_layers=2,
                dropout=dropout,
                non_linearity=:relu,
                batch_size=batch_size,
                seed=model_seed)

            model_path = joinpath(@__DIR__, "../assets/rq_spline_model_hopt_$(model_seed).bson")
            @save model_path model hparams opt
            model_seed += 1

            return model_loss, (hparams.seed, )

        end
    end

    return bohb
end

bohb = run_hyperopt((tres=tres, label=tr_cond_labels, nhits=nhits))


