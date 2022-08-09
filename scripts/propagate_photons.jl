using NeutrinoTelescopes.PhotonPropagationCuda
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.Types
using Logging
using BenchmarkTools
using BenchmarkPlots, StatsPlots
using Plots

#debuglogger = ConsoleLogger(stderr, Logging.Debug)
#global_logger(debuglogger)

n_photons = Int64(1E5)
distance = 25f0
medium = Medium.make_cascadia_medium_properties(Float32)


df, nph_sim = propagate_distance(distance, medium, Int64(ceil(n_photons)))

@df df histogram(:tres, weights=:abs_weight)


suite = BenchmarkGroup()

n_photons = exp10.(4:0.5:9)

for nph in n_photons
    suite[nph] = @benchmarkable $PhotonPropagationCuda.propagate_distance($distance, $medium, Int64(ceil($nph)))
end

tune!(suite)
results = run(suite)

plot(results)

medr = median(results)

scatter(collect(keys(medr)), getproperty.(values(medr), (:time, )) ./ (keys(medr)),
 xscale=:log10, yscale=:log10, ylim=(1E-1, 1E5))

getproperty.(values(medr), (:time, ))