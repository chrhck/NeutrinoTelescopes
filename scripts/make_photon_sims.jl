using Parquet
using DataFrames
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.Modelling
using Logging
using TerminalLoggers

global_logger(TerminalLogger(right_justify=120))

medium = Medium.make_cascadia_medium_properties(Float32)

results_df = make_photon_fits(Int64(1E8), Int64(1E5), 10, 10, 300f0)

write_parquet(joinpath(@__DIR__, "../assets/photon_fits.parquet"), results_df)