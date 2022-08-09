module Utils
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
using DataStructures
using Distributions

export fast_linear_interp, transform_integral_range
export integrate_gauss_quad
export sph_to_cart, rodrigues_rotation
export CategoricalSetDistribution
export sample_cherenkov_track_direction
export rand_gamma

const GL10 = gausslegendre(10)

"""
    fast_linear_interp(x_eval::T, xs::AbstractVector{T}, ys::AbstractVector{T})

Linearly interpolate xs -> ys and evaluate x_eval on interpolation. Assume xs are sorted in ascending order.
"""
function fast_linear_interp(x_eval::T, xs::AbstractVector{T}, ys::AbstractVector{T}) where {T}

    lower = first(xs)
    upper = last(xs)
    x_eval = clamp(x_eval, lower, upper)


    ix_upper = searchsortedfirst(xs, x_eval)
    ix_lower = ix_upper - 1

    @inbounds edge_l = xs[ix_lower]
    @inbounds edge_h = xs[ix_upper]

    step = edge_h - edge_l

    along_step = (x_eval - edge_l) / step

    @inbounds y_low = ys[ix_lower]
    @inbounds slope = (ys[ix_upper] - y_low)

    interpolated = y_low + slope * along_step

    return interpolated

end


function fast_linear_interp(x::T, knots::AbstractVector{T}, lower::T, upper::T) where {T}
    # assume equidistant

    x = clamp(x, lower, upper)
    range = upper - lower
    n_knots = size(knots, 1)
    step_size = range / (n_knots - 1)

    along_range = (x - lower) / step_size
    along_range_floor = floor(along_range)
    lower_knot = Int64(along_range_floor) + 1

    if lower_knot == n_knots
        return @inbounds knots[end]
    end

    along_step = along_range - along_range_floor
    @inbounds y_low = knots[lower_knot]
    @inbounds slope = (knots[lower_knot+1] - y_low)

    interpolated = y_low + slope * along_step

    return interpolated
end


function transform_integral_range(x::Real, f::T, xrange::Tuple{<:Real,<:Real}) where {T<:Function}
    ba_half = (xrange[2] - xrange[1]) / 2
    x = oftype(ba_half, x)

    u_traf = ba_half * x + (xrange[1] + xrange[2]) / 2
    oftype(x, f(u_traf) * ba_half)

end

function integrate_gauss_quad(f::T, a::Real, b::Real) where {T<:Function}
    U = promote_type(typeof(a), typeof(b))
    U(integrate_gauss_quad(f, a, b, GL10[1], GL10[2]))
end

function integrate_gauss_quad(f::T, a::Real, b::Real, order::Integer) where {T<:Function}
    nodes, weights = gausslegendre(order)
    integrate_gauss_quad(f, a, b, nodes, weights)
end

function integrate_gauss_quad(f::T, a::Real, b::Real, nodes::AbstractVector{U}, weights::AbstractVector{U}) where {T<:Function,U<:Real}
    dot(weights, map(x -> transform_integral_range(x, f, (a, b)), nodes))
end

function sph_to_cart(theta::Real, phi::Real)
    sin_theta, cos_theta = sincos(theta)
    sin_phi, cos_phi = sincos(phi)

    T = promote_type(typeof(theta), typeof(phi))
    x::T = cos_phi * sin_theta
    y::T = sin_phi * sin_theta
    z::T = cos_theta

    return SA[x, y, z]
end

"""
CategoricalSetDistribution{T, U<:Real}

Represents a Categorical distribution on a set

### Examples

- `p = CategoricalSetDistribution(Set([:EMinus, :EPlus]), Categorical([0.1, 0.9]))
   rand(p)` -- returns `:EMinus` with 10% probability and `:Eplus` with 90% probability

- `p = CategoricalSetDistribution(Set([:EMinus, :EPlus]), [0.1, 0.9])` -- convenience constructor 
"""
struct CategoricalSetDistribution{T}
    set::OrderedSet{T}
    cat::Categorical

    function CategoricalSetDistribution(set::OrderedSet{T}, cat::Categorical) where {T}
        if length(set) != ncategories(cat)
            error("Set and categorical have to be of same length")
        end
        new{T}(set, cat)
    end

    function CategoricalSetDistribution(set::OrderedSet{T}, probs::Vector{<:Real}) where {T}
        new{T}(set, Categorical(probs))
    end
end

Base.rand(pdist::CategoricalSetDistribution) = pdist.set[rand(pdist.cat)]

"""
    rodrigues_rotation(a, b, operand)

Rodrigues rotation formula. Calculates rotation axis and angle given by rotation a to b.
Apply the resulting rotation to operand.
"""
function rodrigues_rotation(a, b, operand)
    # Rotate a to b and apply to operand
    ax = cross(a, b)
    axnorm = norm(ax)
    ax = ax ./ axnorm
    theta = asin(axnorm)
    rotated = operand .* cos(theta) + (cross(ax, operand) .* sin(theta)) + (ax .* (1 - cos(theta)) .* dot(ax, operand))
    return rotated
end


"""
"""
function sample_cherenkov_track_direction(T::Type)
    # Mystery values from clsim
    angularDist_a = T(0.39) 
    angularDist_b = T(2.61)
    angularDist_I = T(1) - exp(-angularDist_b * 2^angularDist_a)
    
    costheta =  max(T(1) - (-log(T(1) - rand(T)*angularDist_I)/angularDist_b)^(1/angularDist_a), T(-1))
    phi = T(2*π)*rand(T)

    return sph_to_cart(acos(costheta), phi)

end

"""
    rand_gamma(shape, scale)

Sample gamma variates when shape < 1
"""
function rand_gamma(shape, scale)

    d = shape - 1/3 + 1.0
    c = 1.0 / sqrt(9.0 * d)
    κ = d * scale

    rng_val = 0.
    while true
        x = randn()
        v = 1.0 + c * x
        while v <= 0.0
            x = randn()
            v = 1.0 + c * x
        end
        v *= (v * v)
        u = rand()
        x2 = x * x
        if u < 1.0 - 0.331 * abs2(x2) || log(u) < 0.5 * x2 + d * (1.0 - v + log(v))
            rng_val = v*κ
            break
        end
    end

    nia = -1.0 / shape
    randexp = -log(rand())
    rng_val * exp(randexp * nia)

    T = promote_type(shape, scale)
    
    return T(rng_val)

end


end