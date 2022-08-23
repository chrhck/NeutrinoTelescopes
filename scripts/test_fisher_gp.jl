using Distributions
using Plots
using Random
using GaussianProcesses

a = Normal()
b = Exponential()
n = Poisson(20)
x = Uniform(-5, 5)


obs_func(x, a, b) = a*x^3 + b*x^2 + a*b*x
obs_func(x, a) = abs(a)*x^2

function gen_obs(n, a, x)
    nobs = rand(n)
    xs = rand(x, nobs)
    apar = rand(a)
    bpar = rand(a)
    ys = obs_func.(xs, Ref(apar), Ref(bpar))

    xs, ys, apar, bpar
end



obs_x, obs_y, a_par, b_par = gen_obs(n, a, x)

mZero = MeanZero()
kern = Poly(1., 1., 2) +  Mat32Iso(1., 1.)   
logObsNoise = -2.
gp = GP(obs_x, obs_y, mZero, kern)

plot(gp)
optimize!(gp)
plot(gp)

gp


params = []
for i in 1:1000
    obs_x, obs_y, a_par, b_par = gen_obs(n, a, x)

    mZero = MeanZero()
    kern = Mat32Iso(1., 1.)   
    logObsNoise = -2.
    gp = GP(obs_x, obs_y, mZero, kern, logObsNoise)
    try
        optimize!(gp)
        push!(params, [GaussianProcesses.get_params(kern)... a_par b_per])
    catch e
        continue
    end
end

params


fit_params = vcat(params...)
fit_params



scatter(fit_params[:, 3], fit_params[:, 1])
scatter(fit_params[:, 3], fit_params[:, 2])