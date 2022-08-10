using Distributions
using Plots
using NeutrinoTelescopes.Utils

shape = 5.
scale = 2.

g = Gamma(shape, scale)

histogram(rand(g, 100000))

histogram!([rand_gamma(shape, scale) for _ in 1:100000])


@code_lowered rand_gamma(1, 1, Float32)


function cool_gamma(shape, scale)
    d = shape - 1/3
    c = 1/ sqrt(9*d)

    while true

        x = randn()
        v = fma(c, x, one(x))

        if v <= 0
            continue
        end

        v = v * v * v
        u = rand()

        xsq = x^2

        if u < 1 -fma(T(0.0331),xsq^2, one(xsq)) # 1 - 0.0331 * xsq^2
            return d*v*scale
        end

        if log(u) < T(0.5) * xsq + d*(T(1) - v + log(v))
            return d*v*scale
        end
    end
end

histogram!([rand_gamma(shape, scale) for _ in 1:100000])
