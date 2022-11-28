using CUDA
using NNlib
using EllipsisNotation

@inline function my_where(X, A, B)
    broadcast(A, B, X) do a, b, x
        x > 0 ? a : b
    end
end



function _normalize_bin_sizes(unnormalized_bin_sizes,
    total_size,
    min_bin_size)
    """Make bin sizes sum to `total_size` and be no less than `min_bin_size`."""
    num_bins = size(unnormalized_bin_sizes, 1)

    @assert num_bins * min_bin_size <= total_size
    bin_sizes = softmax(unnormalized_bin_sizes)
    return bin_sizes .* (total_size - num_bins * min_bin_size) .+ min_bin_size
end

function _normalize_knot_slopes(unnormalized_knot_slopes,
    min_knot_slope)
    """Make knot slopes be no less than `min_knot_slope`."""
    # The offset is such that the normalized knot slope will be equal to 1
    # whenever the unnormalized knot slope is equal to 0.
    @assert min_knot_slope < 1.0

    offset = log(exp(1.0 - min_knot_slope) - 1.0)
    return softplus(unnormalized_knot_slopes .+ offset) .+ min_knot_slope
end


"""
    constrain_params(params, range_min, range_max, min_bin_size=1e-4, min_knot_slope=1e-4)
Constrain spline parameters.

# Arguments
- params: Array of shape [3 * num_bins + 1, ...]
- range_min: Lower bound of spline range
- range_max: Upper bound of spline range
- min_bin_size: Minimum bin size (used for numerical stability)
- min_knot_slope: Minimum slope at each knot (used for numerical stability)
"""
function constrain_params(params, range_min, range_max, min_bin_size=1e-4, min_knot_slope=1e-4)
    num_bins = div((size(params, 1) - 1), 3)
    unnormalized_bin_widths = params[1:num_bins, ..]
    unnormalized_bin_heights = params[num_bins+1:2*num_bins, ..]
    unnormalized_knot_slopes = params[2*num_bins+1:end, ..]

    # Normalize bin sizes and compute bin positions on the x and y axis.
    range_size = range_max - range_min
    bin_widths = _normalize_bin_sizes(unnormalized_bin_widths, range_size,
        min_bin_size)
    bin_heights = _normalize_bin_sizes(unnormalized_bin_heights, range_size,
        min_bin_size)

    x_pos = range_min .+ cumsum(bin_widths[1:end-1, ..], dims=1)
    y_pos = range_min .+ cumsum(bin_heights[1:end-1, ..], dims=1)
    if ndims(params) == 1
        pad_shape = (1,)
    else
        pad_shape = (1, size(params)[2:end]...)
    end

    pad_below = similar(x_pos, pad_shape)
    pad_below[:] .= range_min
    pad_above = similar(x_pos, pad_shape)
    pad_above[:] .= range_max

    x_pos = vcat(pad_below, x_pos, pad_above)
    y_pos = vcat(pad_below, y_pos, pad_above)
    # Normalize knot slopes and enforce requested boundary conditions.
    knot_slopes = _normalize_knot_slopes(unnormalized_knot_slopes,
        min_knot_slope)

    return x_pos, y_pos, knot_slopes

end

"""Applies a rational-quadratic spline to a scalar.
  Args:
    x: a scalar (0-dimensional array). The scalar `x` can be any real number; it
      will be transformed by the spline if it's in the closed interval
      `[x_pos[0], x_pos[-1]]`, and it will be transformed linearly if it's
      outside that interval.
    x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
    y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
    knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
  Returns:
    A tuple of two scalars: the output of the transformation and the log of the
    absolute first derivative at `x`.
  """
function rqs_univariate(x_pos, y_pos, knot_slopes, x)
    na = [CartesianIndex()]

    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.

    #x_pos = reshape(x_pos, 1, size(x_pos)...)
    #y_pos = reshape(y_pos, 1, size(y_pos)...)
    #knot_slopes = reshape(knot_slopes, 1, size(knot_slopes)...)

    below_range = x .<= x_pos[1, ..]
    above_range = x .>= x_pos[end, ..]

    # this will have shape (n_knots, length(x))
    correct_bin = ((x .>= x_pos[1:end-1, ..]') .&& (x .< x_pos[2:end, ..]'))'

    # is any of the x points in range
    any_bin_in_range = any(correct_bin, dims=1)

    first_bin = falses(size(correct_bin))
    first_bin[1, ..] .= true

    correct_bin = my_where(any_bin_in_range, correct_bin, first_bin)

    x_pos_bin = (x_pos[1:end-1, ..][correct_bin], x_pos[2:end, ..][correct_bin])
    y_pos_bin = (y_pos[1:end-1, ..][correct_bin], y_pos[2:end, ..][correct_bin])
    knot_slopes_bin = (knot_slopes[1:end-1, ..][correct_bin], knot_slopes[2:end, ..][correct_bin])

    bin_width = x_pos_bin[2] .- x_pos_bin[1]
    bin_height = y_pos_bin[2] .- y_pos_bin[1]
    bin_slope = bin_height ./ bin_width

    z = (x .- x_pos_bin[1]) ./ bin_width
    z = clamp.(z, 0.0, 1.0)

    sq_z = z .^ 2
    z1mz = z .- sq_z  # z(1-z)
    sq_1mz = (1.0 .- z) .^ 2
    slopes_term = knot_slopes_bin[2] .+ knot_slopes_bin[1] .- (2.0 .* bin_slope)

    numerator = bin_height .* (bin_slope .* sq_z .+ knot_slopes_bin[1] .* z1mz)
    denominator = bin_slope .+ slopes_term .* z1mz
    y = y_pos_bin[1] .+ numerator ./ denominator


    # Compute log det Jacobian.
    # The logdet is a sum of 3 logs. It is easy to see that the inputs of the
    # first two logs are guaranteed to be positive because we ensured that z is in
    # [0, 1]. This is also true of the log(denominator) because:
    # denominator
    # == bin_slope + (knot_slopes_bin[1] + knot_slopes_bin[0] - 2 * bin_slope) *
    # z*(1-z)
    # >= bin_slope - 2 * bin_slope * z * (1-z)
    # >= bin_slope - 2 * bin_slope * (1/4)
    # == bin_slope / 2
    logdet = 2.0 .* log.(bin_slope) .+ log.(
        knot_slopes_bin[2] .* sq_z .+ 2.0 .* bin_slope .* z1mz .+
        knot_slopes_bin[1] .* sq_1mz) .- 2.0 .* log.(denominator)


    # If x is outside the spline range, we default to a linear transformation.
    y = my_where(below_range, (x .- x_pos[1, ..]) .* knot_slopes[1, ..] .+ y_pos[1, ..], y)
    y = my_where(above_range, (x .- x_pos[end, ..]) .* knot_slopes[end, ..] .+ y_pos[end, ..], y)
    logdet = my_where(below_range, log.(knot_slopes[1, ..]), logdet)
    logdet = my_where(above_range, log.(knot_slopes[end, ..]), logdet)
    return y, logdet
end

function safe_quadratic_root(a, b, c)
    """Implement a numerically stable version of the quadratic formula."""
    # This is not a general solution to the quadratic equation, as it assumes
    # b ** 2 - 4. * a * c is known a priori to be positive (and which of the two
    # roots is to be used, see https://arxiv.org/abs/1906.04032).
    # There are two sources of instability:
    # (a) When b ** 2 - 4. * a * c -> 0, sqrt gives NaNs in gradient.
    # We clip sqrt_diff to have the smallest float number.

    T = promote_type(typeof(a), typeof(b), typeof(c))

    sqrt_diff = b^2 - 4.0 * a * c
    safe_sqrt = sqrt(clamp(sqrt_diff, typemin(T), typemax(T)))

    # If sqrt_diff is non-positive, we set sqrt to 0. as it should be positive.
    safe_sqrt = sqrt_diff > 0 ? safe_sqrt : zero(T)
    # (b) When 4. * a * c -> 0. We use the more stable quadratic solution
    # depending on the sign of b.
    # See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf (eq 7 and 8).
    # Solution when b >= 0
    numerator_1 = 2.0 * c
    denominator_1 = -b - safe_sqrt
    # Solution when b < 0
    numerator_2 = -b + safe_sqrt
    denominator_2 = 2 * a
    # Choose the numerically stable solution.
    numerator = b >= 0 ? numerator_1 : numerator_2
    denominator = b >= 0 ? denominator_1 : denominator_2
    return numerator / denominator

end



function inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)

    below_range = y .<= y_pos[1, ..]
    above_range = y .>= y_pos[end, ..]

    # this will have shape (n_knots, length(x))
    correct_bin = ((y .>= y_pos[1:end-1, ..]') .&& (y .< y_pos[2:end, ..]'))'

    # is any of the x points in range
    any_bin_in_range = any(correct_bin, dims=1)

    first_bin = falses(size(correct_bin))
    first_bin[1, ..] .= true

    correct_bin = my_where(any_bin_in_range, correct_bin, first_bin)

    x_pos_bin = (x_pos[1:end-1, ..][correct_bin], x_pos[2:end, ..][correct_bin])
    y_pos_bin = (y_pos[1:end-1, ..][correct_bin], y_pos[2:end, ..][correct_bin])
    knot_slopes_bin = (knot_slopes[1:end-1, ..][correct_bin], knot_slopes[2:end, ..][correct_bin])

    bin_width = x_pos_bin[2] .- x_pos_bin[1]
    bin_height = y_pos_bin[2] .- y_pos_bin[1]
    bin_slope = bin_height ./ bin_width

    w = (y .- y_pos_bin[1]) ./ bin_height
    w = clamp.(w, 0.0, 1.0)  # Ensure w is in [0, 1].

    # Compute quadratic coefficients: az^2 + bz + c = 0
    slopes_term = knot_slopes_bin[2] .+ knot_slopes_bin[1] .- 2.0 * bin_slope
    c = .-bin_slope .* w
    b = knot_slopes_bin[1] .- slopes_term .* w
    a = bin_slope .- b

    # Solve quadratic to obtain z and then x.
    z = safe_quadratic_root.(a, b, c)
    z = clamp.(z, 0.0, 1.0)  # Ensure z is in [0, 1].
    x = bin_width .* z .+ x_pos_bin[1]

    # Compute log det Jacobian.
    sq_z = z .^ 2
    z1mz = z .- sq_z  # z(1-z)
    sq_1mz = (1.0 .- z) .^ 2
    denominator = bin_slope .+ slopes_term .* z1mz
    logdet = -2.0 .* log.(bin_slope) .- log.(
        knot_slopes_bin[2] .* sq_z .+ 2.0 .* bin_slope .* z1mz .+
        knot_slopes_bin[1] .* sq_1mz) .+ 2.0 .* log.(denominator)

    # If y is outside the spline range, we default to a linear transformation.
    x = my_where(below_range, (y .- y_pos[1, ..]) ./ knot_slopes[1, ..] .+ x_pos[1, ..], x)
    x = my_where(above_range, (y .- y_pos[end, ..]) ./ knot_slopes[end, ..] .+ x_pos[end, ..], x)
    logdet = my_where(below_range, .-log.(knot_slopes[1, ..]), logdet)
    logdet = my_where(above_range, .-log.(knot_slopes[end, ..]), logdet)
    return x, logdet
end

x = -10:0.1:10
num_bins = 5
params = repeat(randn(3 * num_bins + 1), 1, length(x))
x_pos, y_pos, knot_slopes = constrain_params(params, -5, 5)
y, logdet = rqs_univariate(x_pos, y_pos, knot_slopes, x)

xrt, logdet = inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)


all(x .≈ xrt)

lines(x, y[:, 1])



x
x[1:5, :]



begin
    h = randn(10)
    w = sort(randn(10))
    d = randn(10)
    x = randn()

    rqs_univariate(w, h, d, x)
end


h = randn((2, 10))
w = randn((2, 10))
d = randn((2, 9))
x = randn()



rqs_univariate(w[1, :], h[1, :], d[1, :], x[1])


x = randn(10)
y = randn(5, 8)



x .< y[na, ..]
