using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
using BenchmarkTools
using Random
using CategoricalArrays
using AutoMLPipeline
using NeutrinoTelescopes
using StatsBase
using Hyperopt


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

function calc_flow_inputs(p::Particle, target::MultiPMTDetector)

    distance = norm(p.position .- target.position)
    energy = p.energy

    dir_theta, dir_phi = cart_to_sph(p.direction...)
    pos_theta, pos_phi = cart_to_sph(p.position...)

    df_input = DataFrame(
        distance=distance, energy=energy, dir_theta=dir_theta, dir_phi=dir_phi,
        pos_theta=pos_theta, pos_phi=pos_phi)

    return df_input
end


function read_hdf(fnames, nsel_frac=0.8, rng=nothing)

    all_hits = []
    attr_dicts = []
    for fname in fnames
        h5open(fname, "r") do fid
            if !isnothing(rng)
                datasets = shuffle(rng, keys(fid["pmt_hits"]))
            else
                datasets = keys(fid["pmt_hits"])
            end

            if nsel_frac == 1
                index_end = length(datasets)
            else
                index_end = Int(ceil(length(datasets)*nsel_frac))
            end
            

            for grpn in datasets[1:index_end]
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
    end
    
    rnd_ixs = shuffle(1:length(all_hits))

    all_hits = all_hits[rnd_ixs]

    hits_df = reduce(vcat, all_hits)
    tres = hits_df[:, :tres]
    nhits = hits_df[:, :nhits]
    return tres, nhits, preproc_labels(hits_df), DataFrame(attr_dicts)
end

read_hdf(fname::String, nsel, rng) = read_hdf([fname], nsel, rng)

fnames = [
    joinpath(@__DIR__, "../assets/photon_table_extended_2.hd5"),
    joinpath(@__DIR__, "../assets/photon_table_extended_3.hd5"),
    ]
rng = MersenneTwister(31338)
nsel_frac = 0.7
tres, nhits, cond_labels, ds_summary = read_hdf(fnames, nsel_frac, rng)

nrow(cond_labels)

catf = CatFeatureSelector()
ohe = OneHotEncoder()
norm = SKPreprocessor("Normalizer")
numf = NumFeatureSelector()

traf = @pipeline (numf |> norm) + (catf |> ohe)
tr_cond_labels = fit_transform!(traf, cond_labels) |> Matrix |> adjoint

nhits

model, loss = train_model(
        (tres=tres, label=tr_cond_labels, nhits=nhits),
        true,
        epochs=50,
        lr=0.003,
        mlp_layer_size=512,
        mlp_layers=2,
        dropout=0.1,
        non_linearity=:relu,
        batch_size=5000,
        split_final=false,
        rel_weight_poisson=1,
        seed=1)


ho = @hyperopt for i = 25,
    sampler = RandomSampler(),
    batch_size = [1000, 2000, 5000],
    lr = 10 .^ (-3:0.5:-1.5),
    mlp_layer_size = [512, 768, 1024],
    dropout = [0, 0.1, 0.2]

    model, loss = train_model(
        (tres=tres, label=tr_cond_labels, nhits=nhits),
        true,
        epochs=50,
        lr=lr,
        mlp_layer_size=mlp_layer_size,
        dropout=dropout,
        non_linearity=:tanh,
        batch_size=batch_size
    )
    loss
end


h5open(fnames[2], "r") do fid
    Matrix(fid["pmt_hits"]["dataset_200"][:, :])
end



using Flux
Flux.testmode!(model)
h5open(fnames[2], "r") do fid
    df = create_pmt_table(fid["pmt_hits"]["dataset_100"])
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

    t_plot = -5:0.1:100
    l_plot = repeat(Vector(plot_labels_m_tf[1, :]), 1, length(t_plot))


    fig = Figure()
    log_pdf_eval, log_expec = cpu(model)(t_plot, l_plot)

    lines(fig[1, 1], t_plot, exp.(log_pdf_eval))
    hist!(fig[1, 1], plot_tres[mask], bins=-5:3:100, normalization=:pdf)

    lines(fig[1, 2], t_plot, exp.(log_pdf_eval) .* exp.(log_expec))
    hist!(fig[1, 2], plot_tres[mask], bins=-5:3:100)

    fig
end


xs = -50:0.1:50
K = 10
params = zeros((3*K+3), length(xs))
params[end-1, :] .= 0
params[end, :] .= log(3)

fig = Figure()
plot(fig[1, 1], Normal(0, 3), color=:red, lw=3)

plot!(fig[1, 1],xs, exp.(eval_transformed_normal_logpdf(xs, params, -5, 5)))

fig
using Distributions


l_plot = cpu(rq_layer)(t_plot, l_plot)




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
