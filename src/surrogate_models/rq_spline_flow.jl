"""
Implementation of normalizing flows.

Part of this code has been adapted from distrax (see function descriptions)
For these, the following license applies:

# ==============================================================================
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.

"Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

"Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:

    You must give any other recipients of the Work or Derivative Works a copy of this License; and
    You must cause any modified files to carry prominent notices stating that You changed the files; and
    You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and
    If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License.

    You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS
"""
module RQSplineFlow

using CUDA
using Distributions
using NNlib
using ...Utils

export constrain_spline_params, rqs_univariate, inv_rqs_univariate, eval_transformed_normal_logpdf
export sample_flow

@inline function my_where(X, A, B)
    broadcast(A, B, X) do a, b, x
        x > 0 ? a : b
    end
end

@inline function ones_like(::T, size...) where {T}

    if T <: CuArray
        return CuArray(ones(eltype(T), size...))
    else
        return ones(eltype(T), size...)
    end
end

@inline function trues_like(::T, size...) where {T}

    if T <: CuArray
        return CuArray(ones(Bool, size...))
    else
        return trues(size...)
    end
end

@inline function falses_like(::T, size...) where {T}

    if T <: CuArray
        return CuArray(zeros(Bool, size...))
    else
        return falses(size...)
    end
end


"""
    function _normalize_bin_sizes(unnormalized_bin_sizes,
                                total_size,
                                min_bin_size)
Make bin sizes sum to `total_size` and be no less than `min_bin_size`.

Adapted from distrax.
"""
function _normalize_bin_sizes(unnormalized_bin_sizes,
    total_size,
    min_bin_size)

    num_bins = size(unnormalized_bin_sizes, 1)

    @assert num_bins * min_bin_size <= total_size
    bin_sizes = softmax(unnormalized_bin_sizes)
    return bin_sizes .* (total_size - num_bins * min_bin_size) .+ min_bin_size
end


"""
    function _normalize_knot_slopes(unnormalized_knot_slopes,
                                    min_knot_slope)
Make knot slopes be no less than `min_knot_slope`.

Adapted from distrax.
"""
function _normalize_knot_slopes(unnormalized_knot_slopes,
    min_knot_slope)
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

Adapted from distrax.
"""
function constrain_spline_params(params, range_min, range_max, min_bin_size=1e-4, min_knot_slope=1e-4)
    num_bins = div((size(params, 1) - 1), 3)
    unnormalized_bin_widths = params[1:num_bins, :]
    unnormalized_bin_heights = params[num_bins+1:2*num_bins, :]
    unnormalized_knot_slopes = params[2*num_bins+1:end, :]

    # Normalize bin sizes and compute bin positions on the x and y axis.
    range_size = range_max - range_min
    bin_widths = _normalize_bin_sizes(unnormalized_bin_widths, range_size,
        min_bin_size)
    bin_heights = _normalize_bin_sizes(unnormalized_bin_heights, range_size,
        min_bin_size)

    x_pos = range_min .+ cumsum(bin_widths[1:end-1, :], dims=1)
    y_pos = range_min .+ cumsum(bin_heights[1:end-1, :], dims=1)
    if ndims(params) == 1
        pad_shape = (1,)
    else
        pad_shape = (1, size(params)[2:end]...)
    end

    pad_below = ones_like(x_pos, pad_shape)
    pad_below = pad_below .* range_min

    pad_above = ones_like(x_pos, pad_shape)
    pad_above = pad_above .* range_max


    x_pos = vcat(pad_below, x_pos, pad_above)
    y_pos = vcat(pad_below, y_pos, pad_above)
    # Normalize knot slopes and enforce requested boundary conditions.
    knot_slopes = _normalize_knot_slopes(unnormalized_knot_slopes,
        min_knot_slope)

    return x_pos, y_pos, knot_slopes

end

"""
    function safe_quadratic_root(a, b, c)

Implements a numerically stable version of the quadratic formula.

Adapted from distrayx
"""
function safe_quadratic_root(a, b, c)
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


"""
    function rqs_univariate(x_pos, y_pos, knot_slopes, x)
Applies a rational-quadratic spline to a vector.

Implements the spline bijector introduced by:
  > Durkan et al., Neural Spline Flows, https://arxiv.org/abs/1906.04032, 2019.

# Arguments
- x_pos::AbstractMatrix: Bin-boundaries on the x-axis
- y_pos::AbstractMatrix: Bin boundaries on the y-axis
- knot_slopes::AbstractMatrix: Slopes at knot points
- x::AbstractVector: Evalution positions

Adapted from distrax.
"""
function rqs_univariate(x_pos::AbstractMatrix, y_pos::AbstractMatrix, knot_slopes::AbstractMatrix, x::AbstractVector)

    @assert size(x_pos) == size(y_pos) && size(x_pos) == size(knot_slopes)
    @assert size(x_pos, 2) == length(x)

    # Search to find the right bin. NOTE: The bins are sorted

    below_range = x .<= x_pos[1, :]
    above_range = x .>= x_pos[end, :]

    # this will have shape (n_knots, length(x))
    correct_bin = ((x .>= x_pos[1:end-1, :]') .&& (x .< x_pos[2:end, :]'))'

    # is any of the x points in range
    any_bin_in_range = any(correct_bin, dims=1)

    first_bin = vcat(trues_like(x_pos, 1, size(correct_bin, 2)), falses_like(x_pos, size(correct_bin, 1) - 1, size(correct_bin, 2)))

    correct_bin = my_where(any_bin_in_range, correct_bin, first_bin)

    x_pos_bin = (x_pos[1:end-1, :][correct_bin], x_pos[2:end, :][correct_bin])
    y_pos_bin = (y_pos[1:end-1, :][correct_bin], y_pos[2:end, :][correct_bin])
    knot_slopes_bin = (knot_slopes[1:end-1, :][correct_bin], knot_slopes[2:end, :][correct_bin])

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
    y = my_where(below_range, (x .- x_pos[1, :]) .* knot_slopes[1, :] .+ y_pos[1, :], y)
    y = my_where(above_range, (x .- x_pos[end, :]) .* knot_slopes[end, :] .+ y_pos[end, :], y)
    logdet = my_where(below_range, log.(knot_slopes[1, :]), logdet)
    logdet = my_where(above_range, log.(knot_slopes[end, :]), logdet)
    return y, logdet
end


"""
    function inv_rqs_univariate(x_pos::AbstractMatrix, y_pos::AbstractMatrix, knot_slopes::AbstractMatrix, y::AbstractVector)

Inverse of rational-quadratic spline applied to a vector.

Adapted from distrax.
"""
function inv_rqs_univariate(x_pos::AbstractMatrix, y_pos::AbstractMatrix, knot_slopes::AbstractMatrix, y::AbstractVector)

    @assert size(x_pos) == size(y_pos) && size(x_pos) == size(knot_slopes)
    @assert size(x_pos, 2) == length(y)

    below_range = y .<= y_pos[1, :]
    above_range = y .>= y_pos[end, :]

    # this will have shape (n_knots, length(x))
    correct_bin = ((y .>= y_pos[1:end-1, :]') .&& (y .< y_pos[2:end, :]'))'

    # is any of the x points in range
    any_bin_in_range = any(correct_bin, dims=1)

    first_bin = vcat(trues_like(x_pos, 1, size(correct_bin, 2)), falses_like(x_pos, size(correct_bin, 1) - 1, size(correct_bin, 2)))

    correct_bin = my_where(any_bin_in_range, correct_bin, first_bin)

    x_pos_bin = (x_pos[1:end-1, :][correct_bin], x_pos[2:end, :][correct_bin])
    y_pos_bin = (y_pos[1:end-1, :][correct_bin], y_pos[2:end, :][correct_bin])
    knot_slopes_bin = (knot_slopes[1:end-1, :][correct_bin], knot_slopes[2:end, :][correct_bin])

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
    x = my_where(below_range, (y .- y_pos[1, :]) ./ knot_slopes[1, :] .+ x_pos[1, :], x)
    x = my_where(above_range, (y .- y_pos[end, :]) ./ knot_slopes[end, :] .+ x_pos[end, :], x)
    logdet = my_where(below_range, .-log.(knot_slopes[1, :]), logdet)
    logdet = my_where(above_range, .-log.(knot_slopes[end, :]), logdet)
    return x, logdet
end


function _split_params(params)

    spline_params = params[1:end-2, :]
    shift = params[end-1, :]
    scale = sigmoid.(params[end, :]) .* 100
    return spline_params, shift, scale
end

"""
    function eval_transformed_normal_logpdf(y, params, range_min, range_max)

Evaluate logpdf of scaled, shifted, rq-spline applied to normal distribution
"""
function eval_transformed_normal_logpdf(y, params, range_min, range_max)
    @assert length(y) == size(params, 2)
    spline_params, shift, scale = _split_params(params)

    #scale = 5.
    #shift = 0.

    x_pos, y_pos, knot_slopes = constrain_spline_params(spline_params, range_min, range_max)
    x, logdet_spline = inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)


    normal_logpdf = -0.5 .* (x .^ 2 .+ log(2 * pi))

    normal_logpdf = -log.(scale) .- 0.5 .* (log(2 * π) .+ ((x .- shift) ./ scale) .^ 2)
    #normal_logpdf =  logpdf.(Normal(0, 1), x)

    return normal_logpdf .+ logdet_spline
end


function sample_flow(params, range_min, range_max, n_per_param)

    @assert length(n_per_param) == size(params, 2)

    param_vec = repeat_for(params, n_per_param)

    spline_params, shift, scale = _split_params(param_vec)
    x_pos, y_pos, knot_slopes = constrain_spline_params(spline_params, range_min, range_max)
    x = randn(size(param_vec, 2)) .* scale .+ shift
    y, _ = rqs_univariate(x_pos, y_pos, knot_slopes, x)
    return y
end



end
