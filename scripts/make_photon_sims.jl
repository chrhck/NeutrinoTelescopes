using Parquet
using DataFrames
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.Modelling

medium = Medium.make_cascadia_medium_properties(Float32)

results_df = make_photon_fits(Int64(1E8), Int64(1E5), 250, 250, 300f0)

@__DIR__
write_parquet(joinpath(@__DIR__, "../assets/photon_fits.parquet"), results_df)