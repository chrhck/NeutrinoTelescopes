
using Distributions
using Plots
g = Gamma(1.1, 1.5)

histogram(rand(g, 100000))



histogram!([rand_gamma(1.1, 1.5) for _ in 1:100000])
