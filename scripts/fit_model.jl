using Plots
using StatsPlots
using Parquet
using Hyperopt
using Random
using BSON: @save
using Flux

using NeutrinoTelescopes.Modelling

infile = joinpath(@__DIR__, "../assets/photon_fits.parquet")
seed = 31337
epochs = 500

n_it = 50
ho = @hyperopt for i = n_it,
    sampler = CLHSampler(dims=[Categorical(3), Continuous(), Categorical(5), Continuous()]),# This is default if none provided
    batch_size = [2048, 4096, 8192],
    dropout_rate = LinRange(0, 0.3, n_it),
    width = [64, 128, 256, 512, 1024],
    learning_rate = 10 .^ LinRange(-4, -2, n_it)

    lr_sched_pars = NoSchedulePars(learning_rate)
    model, test_data = train_mlp(
        epochs=epochs, width=width, lr_schedule_pars=lr_sched_pars, batch_size=batch_size, data_file=infile,
        dropout_rate=dropout_rate, seed=seed)
    print("Epochs: $epochs, Width: $width, lr_sched: $lr_sched_pars, batch: $batch_size, dropout: $dropout_rate ")
    @show loss_all(test_data, model)
end
plot(ho, yscale=:log10)
min_pars = ho.minimizer

params = Dict(
		:width=>min_pars[3],
		:lr_schedule_pars=> NoSchedulePars(min_pars[4]),
		:batch_size=>min_pars[1],
		:data_file=>infile,
		:dropout_rate=>min_pars[2],
		:seed=>31138,
		:epochs=>epochs,
		)


model, _ = train_mlp(;params...)
model = cpu(model)
@save joinpath(@__DIR__, "../assets/photon_model.bson") model params




n_it = 10
ho = @hyperopt for i = n_it,
    sampler = CLHSampler(dims=[Categorical(3), Continuous(), Categorical(5), Continuous(), Continuous(), Categorical(4)]),# This is default if none provided
    batch_size = [2048, 4096, 8192],
    dropout_rate = LinRange(0, 0.3, n_it),
    width = [64, 128, 256, 512, 1024],
    lr_max = 10 .^ LinRange(-4, -2, n_it),
    lr_ratio = 10 .^ LinRange(-3, -1, n_it),
    period = [10, 20, 50, 100]

    lr_min = lr_ratio * lr_max
    lr_sched_pars = SinDecaySchedulePars(lr_min=lr_min, lr_max=lr_max, lr_period=period)
    model, test_data = train_mlp(
        epochs=epochs, width=width, lr_schedule_pars=lr_sched_pars, batch_size=batch_size, data_file=infile,
        dropout_rate=dropout_rate, seed=seed)
    print("Epochs: $epochs, Width: $width, lr_sched: $lr_sched_pars, batch: $batch_size, dropout: $dropout_rate ")
    @show loss_all(test_data, model)
end

plot(ho, yscale=:log10)

min_pars = ho.minimizer
lr_ratio = min_pars[5]
lr_max = min_pars[4]
lr_min = lr_ratio * lr_max
lr_period = min_pars[6]

params = Dict(
		:width=>min_pars[3],
		:lr_schedule_pars=>  SinDecaySchedulePars(lr_min=lr_min, lr_max=lr_max, lr_period=lr_period),
		:batch_size=>min_pars[1],
		:data_file=>infile,
		:dropout_rate=>min_pars[2],
		:seed=>31138,
		:epochs=>epochs,
		)


model, _ = train_mlp(;params...)
model = cpu(model)
@save joinpath(@__DIR__, "../assets/photon_model.bson") model params