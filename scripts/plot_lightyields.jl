using Plots
using Traceur
using NeutrinoTelescopes
using NeutrinoTelescopes.Spectral
using NeutrinoTelescopes.Medium
using NeutrinoTelescopes.LightYield
using NeutrinoTelescopes.Emission
using NeutrinoTelescopes.Utils
using StaticArrays


log_energies = 2:0.1:8
zs = (0:0.1:30.0)# m
medium = make_cascadia_medium_properties(Float64)

# Plot longitudinal profile
plot(zs, longitudinal_profile.(Ref(1E3), zs, Ref(medium), Ref(LightYield.EMinus)), label="1E3 GeV", title="Longitudinal Profile")
plot!(zs, longitudinal_profile.(Ref(1E5), zs, Ref(medium), Ref(LightYield.EMinus)), label="1E5 GeV")

radiation_length(medium)
Medium.density(medium)

# Show fractional contribution for a segment of shower depth
frac_contrib = fractional_contrib_long(1E5, zs, medium, LightYield.EMinus)
plot(zs, frac_contrib, linetype=:steppost, label="", ylabel="Fractional light yield")
sum(frac_contrib)

ftamm_norm = frank_tamm_norm((200, 800), wl -> get_refractive_index(wl, medium))
light_yield = cherenkov_track_length.(1E5, LightYield.EMinus)

plot(zs, frac_contrib .* light_yield, linetype=:steppost, label="", ylabel="Light yield per segment")

# Calculate Cherenkov track length as function of energy 
tlens = cherenkov_track_length.((10 .^ log_energies), LightYield.EMinus)
plot(log_energies, tlens, yscale=:log10, xlabel="Log10(E/GeV)", ylabel="Cherenkov track length")

total_lys = frank_tamm_norm((200.0, 800.0), wl -> get_refractive_index(wl, medium)) * tlens

plot(log_energies, total_lys, yscale=:log10, ylabel="Number of photons", xlabel="log10(Energy/GeV)")


nodes = 5:100

norms = [frank_tamm_norm((200.0, 800.0), wl -> get_refractive_index(wl, medium), n) for n in nodes]
plot(nodes, norms)

@trace cherenkov_track_length.((10 .^ log_energies), LightYield.EMinus)

lambdas = 200:1.:800

plot(lambdas, get_refractive_index.(lambdas, Ref(medium)))
plot(lambdas, get_dispersion.(lambdas, Ref(medium)))


hobo_diff = diff(get_refractive_index.(lambdas, Ref(medium)))
plot!(lambdas[2:end], hobo_diff)


plot(group_velocity.(lambdas, Ref(medium)))

angularDist_a = 0.39 
angularDist_b = 2.61

angularDist_I = 1. - exp(-angularDist_b * 2^angularDist_a)

genf = x -> max(1. - (-log(1. - x*angularDist_I)/angularDist_b)^(1/angularDist_a), -1.)


n_vals = 100000

cosvals = genf.(rand(n_vals))
phi_vals = 2*π*rand(n_vals)

rot_dir = sph_to_cart.(acos.(cosvals), phi_vals)

ph_phi = 2*π*rand(n_vals)
ph_theta = fill(deg2rad(42), n_vals)

ph_dir = sph_to_cart.(ph_theta, ph_phi)


rot_vecs = rodrigues_rotation.(fill(SA[0., 0., 1.], n_vals), rot_dir, ph_dir)

[rv[3] for rv in rot_vecs]

histogram(([(rv[3]) for rv in rot_vecs]), normalize=true)

angls = -1:0.01:1

plot!(angls, cherenkov_ang_dist.(angls, 1.35) / cherenkov_ang_dist_int(1.35))



const double cos_val=std::max(1.-std::pow(-std::log(1.-mwcRngRandomNumber_co(rngState, rngA)*angularDist_I_)/angularDist_b_, one_over_angularDist_a_), -1.);