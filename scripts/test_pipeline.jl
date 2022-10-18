using NeutrinoTelescopes
using Plots
using StaticArrays
using Random
using DataFrames
using StatsPlots
using Distributions
using Parquet
using Dagger
using DTables
using BenchmarkTools
using CUDA


hits_test = DataFrame(time=[0, 0, 0], pmt_id=[1, 1, 2], total_weight=[0, 0, 0])
resample_simulation(hits_test)




distance = 20f0
medium = make_cascadia_medium_properties(Float32)
pmt_area=Float32((75e-3 / 2)^2*π)
target_radius = 0.21f0
target = MultiPMTDetector(@SVector[distance, 0f0, 0f0], target_radius, pmt_area, 
    make_pom_pmt_coordinates(Float32))

zenith_angle = 20f0
azimuth_angle = 10f0


particle = Particle(
        @SVector[0.0f0, 0f0, 0.0f0],
        sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
        0f0,
        Float32(1E5),
        PEMinus
)

upsampling_factor = 10

prop_source = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0), upsampling_factor)

res = Dagger.@spawn propagate_photons(prop_source, target, medium)
hits = Dagger.@spawn make_hits_from_photons(res, prop_source, target, medium, SA[0., 0., 1.])

resampled = Dagger.@spawn resample_simulation(hits, upsampling_factor, true)

resampled = fetch(resampled)

resampled = subtract_mean_tt!(apply_tt!(resampled, STD_PMT_CONFIG.tt_dist), STD_PMT_CONFIG.tt_dist)





thunks = []
for grp in groupby(resampled, :pmt_id)
    ps = Dagger.@spawn PulseSeries(grp, STD_PMT_CONFIG.spe_template, STD_PMT_CONFIG.pulse_model)
    wf = Dagger.@spawn digitize_waveform(
        ps,
        STD_PMT_CONFIG.sampling_freq,
        STD_PMT_CONFIG.adc_freq,
        STD_PMT_CONFIG.noise_amp,
        STD_PMT_CONFIG.lp_filter)
    unfolded = Dagger.@spawn unfold_waveform(wf, STD_PMT_CONFIG.pulse_model_filt, STD_PMT_CONFIG.unf_pulse_res, 0.2, :fnnls)

    Dagger.@spawn do 

    push!(thunks, unfolded)
end


fetch.(thunks)



PulseSeries(_, pmt_config.spe_template, pmt_config.pulse_model) |>
      digitize_waveform(
        _,
        pmt_config.sampling_freq,
        pmt_config.adc_freq,
        pmt_config.noise_amp,
        pmt_config.lp_filter
      ) |>
      unfold_waveform(_, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.2, :fnnls)




thunks = [Dagger.@spawn apply_tt!(group, STD_PMT_CONFIG.tt_dist)]


for group in groupby(resampled, :pmt_id)
    
end