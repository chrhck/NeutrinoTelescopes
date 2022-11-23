using NeutrinoTelescopes
using Logging
using BenchmarkTools
using BenchmarkPlots, StatsPlots
using Plots
using StaticArrays
using CUDA
using StructArrays
#debuglogger = ConsoleLogger(stderr, Logging.Debug)
#global_logger(debuglogger)


distance = 80.0f0
medium = make_cascadia_medium_properties(0.99f0)
n_pmts = 16
pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0

suite = BenchmarkGroup()
n_photons = exp10.(5:0.5:11)
target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area, UInt16(1))
target2 = DetectionSphere(@SVector[0.0f0, 5.0f0, distance], target_radius, n_pmts, pmt_area, UInt16(2))



spectrum = CherenkovSpectrum((300.0f0, 800.0f0), 30, medium)
nph = 1E9
source = PointlikeIsotropicEmitter(SA[0.0f0, 0.0f0, 0.0f0], 0.0f0, Int64(ceil(nph)))



function run_old(source, target, medium, spectrum)
    NeutrinoTelescopes.PhotonPropagationCuda.run_photon_prop_no_local_cache(
        [source], target, medium, spectrum; time_type=Float32)
    nothing
end

function run_new(source, target, medium, spectrum)
    NeutrinoTelescopes.PhotonPropagationCuda.run_photon_prop_no_local_cache(
        [source], [target, target2], medium, spectrum; time_type=Float32)
    nothing
end

suite = BenchmarkGroup()
suite["old"] = begin
    CUDA.@sync @benchmarkable $run_old($source, $target, $medium, $spectrum)
end

suite["new"] = begin
    CUDA.@sync @benchmarkable $run_new($source, $target, $medium, $spectrum)
end
tune!(suite)
results = run(suite, seconds=20)
plot(results)


suite = BenchmarkGroup()
for nph in n_photons
    source = PointlikeIsotropicEmitter(SA[0.0f0, 0.0f0, 0.0f0], 0.0f0, Int64(ceil(nph)))
    suite[nph] = CUDA.@sync @benchmarkable $propagate_photons($source, $target, $medium, $spectrum)
end

tune!(suite)
results = run(suite, seconds=20)

plot(results)

medr = median(results)

p = scatter(collect(keys(medr)), getproperty.(values(medr), (:time,)) ./ (keys(medr)),
    xscale=:log10, yscale=:log10, ylim=(1E-1, 1E5),
    xlabel="Number of Photons", ylabel="Time per Photon (ns)",
    label="", dpi=150, title=CUDA.name(CUDA.device()))

savefig(p, joinpath(@__DIR__, "../figures/photon_benchmark.png"),)

#=
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
=#
