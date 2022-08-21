using Parquet
using Plots
using StatsPlots

results_df = read_parquet(joinpath(@__DIR__, "../assets/photon_fits.parquet"))

@df results_df corrplot(cols(1:5))
@df results_df scatter(:distance, :det_fraction, yscale=:log10)


