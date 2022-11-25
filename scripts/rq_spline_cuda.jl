using CUDA
using NNlib

@inline function my_where(X, A, B)
    broadcast(A, B, X) do a, b, x
        x > 0 ? a : b
    end
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

    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.
    below_range = x .<= x_pos[1]
    above_range = x .>= x_pos[end]
    correct_bin = (x .>= x_pos[1:end-1]) .&& (x .< x_pos[2:end])

    any_bin_in_range = any(correct_bin)
    first_bin = vcat(ones(Bool, (1,)), zeros(Bool, length(correct_bin) - 1))

    correct_bin = my_where(any_bin_in_range, correct_bin, first_bin)

    x_pos_bin = (x_pos[1:end-1][correct_bin], x_pos[2:end][correct_bin])
    y_pos_bin = (y_pos[1:end-1][correct_bin], y_pos[2:end][correct_bin])
    knot_slopes_bin = (knot_slopes[1:end-1][correct_bin], knot_slopes[2:end][correct_bin])

    bin_width = x_pos_bin[2] - x_pos_bin[1]
    bin_height = y_pos_bin[2] - y_pos_bin[1]
    bin_slope = bin_height / bin_width

    z = (x .- x_pos_bin[1]) ./ bin_width

    z = clamp.(z, 0.0, 1.0)
    sq_z = z .^ 2
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 .- z) .^ 2
    slopes_term = knot_slopes_bin[2] .+ knot_slopes_bin[1] .- 2.0 .* bin_slope
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
    logdet = 2.0 .* log(bin_slope) .+ log(
        knot_slopes_bin[2] .* sq_z .+ 2.0 .* bin_slope .* z1mz .+
        knot_slopes_bin[1] .* sq_1mz) .- 2.0 .* log(denominator)

    # If x is outside the spline range, we default to a linear transformation.
    y = my_where(below_range, (x - x_pos[1]) * knot_slopes[1] + y_pos[1], y)
    y = my_where(above_range, (x - x_pos[end]) * knot_slopes[end] + y_pos[end], y)
    logdet = my_where(below_range, log(knot_slopes[1]), logdet)
    logdet = my_where(above_range, log(knot_slopes[end]), logdet)
    return y, logdet
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

function constrain_params(params, range_min, range_max, min_bin_size=1e-4, min_knot_slope=1e-4)
    num_bins = div((size(params, 1) - 1), 3)
    @show num_bins
    unnormalized_bin_widths = params[1:num_bins]
    unnormalized_bin_heights = params[num_bins+1:2*num_bins]
    unnormalized_knot_slopes = params[2*num_bins+1:end]

    # Normalize bin sizes and compute bin positions on the x and y axis.
    range_size = range_max - range_min
    bin_widths = _normalize_bin_sizes(unnormalized_bin_widths, range_size,
        min_bin_size)
    bin_heights = _normalize_bin_sizes(unnormalized_bin_heights, range_size,
        min_bin_size)

    x_pos = range_min .+ cumsum(bin_widths[1:end-1])
    y_pos = range_min .+ cumsum(bin_heights[1:end-1])
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

num_bins = 5
x_pos, y_pos, knot_slopes = constrain_params(randn(3 * num_bins + 1), -5, 5)

x = randn()
rqs_univariate(x_pos, y_pos, knot_slopes, x)


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
