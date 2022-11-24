using Bijectors
using HDF5
using DataFrames
using CairoMakie
using Flux
using Optim
using Base.Iterators
using NNlib
using ProgressLogging
using MLUtils
using CUDA
using BenchmarkTools
using Random
using CategoricalArrays
using AutoMLPipeline
using NeutrinoTelescopes
using TensorBoardLogger
using Logging
using SpecialFunctions


device = cpu


function create_pmt_table(grp, limit=true)
    hits = DataFrame(grp[:, :], [:tres, :pmt_id])
    for (k, v) in attrs(grp)
        hits[!, k] .= v
    end

    hits[!, :nhits] .= nrow(hits)

    if limit && (nrow(hits) > 200)
        hits = hits[1:200, :]
    end

    return hits
end


function preproc_labels(df)
    
    df[!, :log_distance] = log.(df[:, :distance])
    df[!, :log_energy] = log.(df[:, :energy])
    dir_cart = DataFrame(reduce(hcat, sph_to_cart.(df[:, :dir_theta], df[:, :dir_phi]))', [:dir_x, :dir_y, :dir_z])
    pos_cart = DataFrame(reduce(hcat, sph_to_cart.(df[:, :pos_theta], df[:, :pos_phi]))', [:pos_x, :pos_y, :pos_z])

    if "pmt_id" in names(df)  
        df[!, :pmt_id] = categorical(df[:, :pmt_id], levels=1:16)
        feat = [:pmt_id, :log_distance, :log_energy]
    else
        feat = [:log_distance, :log_energy]
    end

    cond_labels = hcat(df[:, feat], dir_cart, pos_cart)
    return cond_labels
end


function read_hdf(fname, nsel=500, rng=nothing)

    all_hits = []
    attr_dicts = []
    h5open(fname, "r") do fid
        if !isnothing(rng)
            datasets = shuffle(rng, keys(fid["pmt_hits"]))
        else
            datasets = keys(fid["pmt_hits"])
        end

        
        for grpn in datasets[1:nsel]
            grp = fid["pmt_hits"][grpn]
            hits = create_pmt_table(grp)
            push!(all_hits, hits)
        end

       
        for grpn in datasets
            grp = fid["pmt_hits"][grpn]

            if length(grp) < 5
                continue
            end

            att_d = Dict(attrs(grp))
            att_d["nhits"] = length(grp)
            att_d["hittime_mean"] = mean(grp[:, 1])
            att_d["hittime_std"] = std(grp[:, 1])

            push!(attr_dicts, att_d)        
        end        
    end

    hits_df = reduce(vcat, all_hits)
    tres = hits_df[:, :tres]
    nhits = hits_df[:, :nhits]
    return tres, nhits, preproc_labels(hits_df), DataFrame(attr_dicts)

end

fname = joinpath(@__DIR__, "../assets/photon_table_extended_2.hd5")
rng = MersenneTwister(31338)
nsel = 15000

tres, nhits, cond_labels, ds_summary = read_hdf(fname, nsel, rng)


function train_model!(loss, opt, train, test, pars, epochs, logger)
    local train_loss
    @progress for epoch in 1:epochs
        total_loss = 0
        for d in train
            # d = d |> device
            gs = gradient(pars) do
                train_loss = loss(d)
                return train_loss
            end
            total_loss += train_loss /
            #=
            with_logger(logger) do
                @info "train" batch_loss=train_loss
            end
            =#
            Flux.update!(opt, pars, gs)
        end
        total_loss /= numobs(train)

        total_loss_test = 0
        for d in test
            total_loss_test += loss(d)
        end
        total_loss_test /= numobs(test)


        with_logger(logger) do
            @info "loss" train=total_loss test=total_loss_test
            
        end
        println("Epoch: $epoch, Train: $total_loss Test: $total_loss_test")

    end
end



summary_data = ds_summary[:, [:nhits, :hittime_mean, :hittime_std]]
summary_labels = preproc_labels(ds_summary)

device = gpu

norm_sum_lab = SKPreprocessor("Normalizer")
norm_targ = SKPreprocessor("Normalizer")

tr_summary_labels = fit_transform!(norm_sum_lab, summary_labels) |> Matrix |> adjoint
tr_summary_data = fit_transform!(norm_targ, summary_data) |> Matrix |> adjoint
for compression in [2, 4, 8, 512]

    pre_model = Chain(
        Dense(ncol(summary_labels) => 512, relu),
        Dense(512 => compression, relu),
        Dense(compression => 512, relu),
        Dense(512 => ncol(summary_data))
        ) |> device


    mse_loss(x) = Flux.Losses.mse(pre_model(x[:label]), x[:data])

    data = DataLoader(
        (data=tr_summary_data, label=tr_summary_labels),
        batchsize=100,
        shuffle=true,
        rng=rng)


    pars = Flux.params(pre_model)
    epochs = 150
    lg = TBLogger("tensorboard_logs/pre_model_comp_$compression", tb_overwrite)
    opt = Flux.Optimise.Adam(0.001)
    train_model!(mse_loss, opt, data, pars, epochs, lg)
end



catf = CatFeatureSelector()
ohe = OneHotEncoder()
norm = SKPreprocessor("Normalizer")
numf = NumFeatureSelector()

traf = @pipeline (numf |> norm) + (catf |> ohe)
tr_cond_labels = fit_transform!(traf, cond_labels) |> Matrix |> adjoint

struct RQNormFlow
    embedding::Chain
    K::Integer
    B::Integer
end

Flux.@functor RQNormFlow (embedding,)

function RQNormFlow(K, B, hidden_structure)

    model = []
    push!(model, Dense(24 => hidden_structure[1], relu))
    for ix in 2:length(hidden_structure)
        push!(model, Dense(hidden_structure[ix-1] => hidden_structure[ix], relu))
    end
    # 3 K -1 for spline, 1 for shift, 1 for scale, 1 for log-expectation
    push!(model, Dense(hidden_structure[end] => 3 * K + 2))

    return RQNormFlow(Chain(model...), K, B)
end


function (m::RQNormFlow)(x, cond)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.

    base = Normal()
    params = m.embedding(Float64.(cond))

    
    logpdf_eval = CUDA.@allowscalar @inbounds [
        logpdf(
            transformed(
                base,
                Bijectors.Scale(sigmoid(p[end-1]) * 50)
                ∘ Bijectors.Shift(p[end-2]) 
                ∘ Bijectors.RationalQuadraticSpline(
                    p[1:m.K],
                    p[m.K+1:2*m.K],
                    p[2*m.K+1:end-3],
                    m.B
                )),
            ax)
        for (p, ax) in zip(eachcol(params), x)
    ]

    log_expec = params[end, :]
    
    return logpdf_eval, log_expec

end


B = 5
K = 15
device = cpu


train_data, test_data = splitobs((tres=tres, label=tr_cond_labels, nhits=nhits); at=0.85)

train_loader = DataLoader(
    train_data,
    batchsize=3000,
    shuffle=true,
    rng=rng)

test_loader = DataLoader(
    test_data,
    batchsize=10000,
    shuffle=true,
    rng=rng)


rq_layer = RQNormFlow(K, B, [512, 512]) |> device
function combined_loss(x)
    logpdf_eval, log_expec = rq_layer(x[:tres], x[:label])

    # poisson: log(exp(-lambda) * lambda^k)
    poiss_f = -exp.(log_expec) .+ x[:nhits] .* log_expec .-loggamma.(x[:nhits] .+1)

    return -sum(logpdf_eval) - sum(poiss_f) 
end 

pars = Flux.params(rq_layer)
opt = Flux.Optimise.Adam(0.001)
epochs = 50

lg = TBLogger("tensorboard_logs/nflow", tb_overwrite)


train_model!(combined_loss, opt, train_loader, test_loader, pars, epochs, lg)
h5open(fname, "r") do fid
    df = create_pmt_table(fid["pmt_hits"]["dataset_10700"])
    plot_tres = df[:, :tres]
    plot_labels = preproc_labels(df)
    @show combine(groupby(plot_labels, :pmt_id), nrow => :n)

    n_per_pmt = combine(groupby(plot_labels, :pmt_id), nrow => :n)
    max_i = argmax(n_per_pmt[:, :n])


    pmt_plot = n_per_pmt[max_i, :pmt_id]
    mask = plot_labels.pmt_id.==pmt_plot

    plot_tres_m = plot_tres[mask]
    plot_labels_m = plot_labels[mask, :]

    plot_labels_m_tf = AutoMLPipeline.transform(traf, plot_labels_m)

    t_plot = -5:0.1:100
    l_plot = repeat(Vector(plot_labels_m_tf[1, :]), 1, length(t_plot))

    fig = Figure()
    log_pdf_eval, log_expec = rq_layer(t_plot, l_plot)
    lines(fig[1, 1], t_plot, exp.(log_pdf_eval))
    hist!(fig[1, 1], plot_tres[mask], bins=-5:3:100, normalization=:pdf)

    lines(fig[1, 2], t_plot, exp.(log_pdf_eval) .* exp.(log_expec))
    hist!(fig[1, 2], plot_tres[mask], bins=-5:3:100)

    fig
end











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
