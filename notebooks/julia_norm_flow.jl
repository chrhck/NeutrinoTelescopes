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


device = cpu


fname = joinpath(@__DIR__, "../assets/photon_table_extended_2.hd5")
fid = h5open(fname, "r")

rng = MersenneTwister(31338)
datasets = shuffle(rng, keys(fid["pmt_hits"]))
nsel = 500

function create_pmt_table(grp)
    hits = DataFrame(grp[:, :], [:tres, :pmt_id])
    for (k, v) in attrs(grp)
        hits[!, k] .= v
    end
    return hits
end


all_hits = []
for grpn in datasets[1:nsel]
    grp = fid["pmt_hits"][grpn]
    hits = create_pmt_table(grp)
    push!(all_hits, hits)
end


function preproc(df)
    tres = df[:, :tres]

    df[!, :pmt_id] = categorical(df[:, :pmt_id], levels=1:16)
    df[!, :log_distance] = log.(df[:, :distance])
    df[!, :log_energy] = log.(df[:, :energy])
    dir_cart = DataFrame(reduce(hcat, sph_to_cart.(df[:, :dir_theta], df[:, :dir_phi]))', [:dir_x, :dir_y, :dir_z])
    pos_cart = DataFrame(reduce(hcat, sph_to_cart.(df[:, :pos_theta], df[:, :pos_phi]))', [:pos_x, :pos_y, :pos_z])

    feat = [:pmt_id, :log_distance, :log_energy]
    cond_labels = hcat(df[:, feat], dir_cart, pos_cart)
    return tres, cond_labels
end


tres, cond_labels = preproc(reduce(vcat, all_hits))

extrema(tres)

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
    push!(model, Dense(hidden_structure[end] => 3 * K))

    return RQNormFlow(Chain(model...), K, B)
end


function (m::RQNormFlow)(x, cond)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.

    base = Normal()
    params = m.embedding(Float64.(cond))
    
    res = CUDA.@allowscalar @inbounds [
        logpdf(
            transformed(
                base,
                Bijectors.Scale(sigmoid(p[end]) * 50) ∘ 
                Bijectors.RationalQuadraticSpline(
                    p[1:m.K],
                    p[m.K+1:2*m.K],
                    p[2*m.K+1:end-1],
                    m.B
                )),
            ax)
        for (p, ax) in zip(eachcol(params), x)
    ]

    return res

end

function train_model!(loss, data, pars, epochs, logger)
    local train_loss
    @progress for epoch in 1:epochs
        total_loss = 0
        for d in data
            gs = gradient(pars) do
                train_loss = loss(d)
                return train_loss
            end
            total_loss += train_loss
            with_logger(logger) do
                @info "train" batch_loss=train_loss
            end
            Flux.update!(opt, pars, gs)
        end
        total_loss /= length(data)
        with_logger(logger) do
            @info "train" loss=total_loss
        end
        println("Epoch: $epoch, Loss: $total_loss")

    end
end


B = 5
K = 15



device = cpu

size(tres)
size(tr_cond_labels)


data = DataLoader(
    (tres=tres |> device, label=tr_cond_labels |> device),
    batchsize=5000,
    shuffle=true,
    rng=rng)
rq_layer = RQNormFlow(K, B, [512, 512]) |> device


loss(x) = -sum(rq_layer(x[:tres], x[:label]))

pars = Flux.params(rq_layer)
opt = Flux.Optimise.Adam(0.001)
epochs = 10

lg = TBLogger("tensorboard_logs/run", tb_overwrite)


train_model!(loss, data, pars, epochs, lg)

plot_tres, plot_labels = preproc(create_pmt_table(fid["pmt_hits"]["dataset_800"]))


combine(groupby(plot_labels, :pmt_id), nrow => :n)


pmt_plot = 12
mask = plot_labels.pmt_id.==pmt_plot

plot_tres_m = plot_tres[mask]
plot_labels_m = plot_labels[mask, :]

plot_labels_m_tf = AutoMLPipeline.transform(traf, plot_labels_m)

t_plot = -5:0.1:50
l_plot = repeat(Vector(plot_labels_m_tf[1, :]), 1, length(t_plot))

fig = Figure()

lines(fig[1, 1], t_plot, exp.(rq_layer(t_plot, l_plot)))

hist!(fig[1, 1], plot_tres[mask], bins=-5:0.5:50, normalization=:pdf)
fig







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
