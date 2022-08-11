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
distance = 80f0
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




log_energies = 2:0.5:5.5

suite = BenchmarkGroup()

for log_energy in log_energies

    particle = Particle(
            @SVector[0.0f0, 0.0f0, 0.0f0],
            @SVector[0.0f0, 0.0f0, 1.0f0],
            0f0,
            Float32(10^log_energy),
            PEMinus
    )
    
    source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
    
    ppcu.initialize_photon_state(source, medium)
    
    distance = 50f0
    n_pmts=16
    pmt_area=Float32((75e-3 / 2)^2*π)
    target_radius = 0.21f0
    
    target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)
 
    bench = @benchmarkable $ppcu.propagate_photons($source, $target, $medium, 512, 92, Int32(100000))

    suite[source.photons] = bench
end

tune!(suite)
results = run(suite)

plot(results)

medr = median(results)

scatter(collect(keys(medr)), getproperty.(values(medr), (:time, )) ./ (keys(medr)),
 xscale=:log10, yscale=:log10, ylim=(1E-1, 1E5))
