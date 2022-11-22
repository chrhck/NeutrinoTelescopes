using CUDA

function rqs_univariate(widths, heights, derivatives, x::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    # We're working on [-B, B] and `widths[end]` is `B`
    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return one(T) * x
    end

    K = length(widths)

    # Find which bin `x` is in; subtract 1 because `searchsortedfirst` returns idx of ≥ not ≤
    k = searchsortedfirst(widths, x) - 1

    # Width
    # If k == 0 then we should put it in the bin `[-B, widths[1]]`
    w_k = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - w_k

    # Slope
    h_k = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - h_k

    s = Δy / w
    ξ = (x - w_k) / w

    # Derivatives at knot-points
    # Note that we have (K - 1) knot-points, not K
    d_k = (k == 0) ? one(T) : derivatives[k]
    d_kplus1 = (k == K - 1) ? one(T) : derivatives[k + 1]

    # Eq. (14)
    numerator = Δy * (s * ξ^2 + d_k * ξ * (1 - ξ))
    denominator = s + (d_kplus1 + d_k - 2s) * ξ * (1 - ξ)
    g = h_k + numerator / denominator

    return g
end


function rqs_univariate(widths, heights, derivatives, x::CuVector{<:Real})
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))


    output = CUDA.ones(T, size(x))
    outmask = (x .≤ -widths[:, end]) .|| (x .≥ widths[:, end])
    output[outmask] .*= x[outmask]

    inmask = .!outmask
    widths_in = widths[inmask]
    heights_in = heights[inmask]
    x_in = x[inmask]

    K = size(widths_in, 1)

    # Find which bin `x` is in; subtract 1 because `searchsortedfirst` returns idx of ≥ not ≤
    k = searchsortedfirst.(widths_in, x_in) .- 1

    # Width
    # If k == 0 then we should put it in the bin `[-B, widths_in[1]]`
    
    mask = k .!= 0

    w_k = -widths_in[:, end]
    w_k[mask] = widths_in[k]
    
    w = widths_in[:, k .+ 1] .- w_k

    # Slope    
    h_k = -heights_in[:, end]
    h_k[mask] = heights_in[k]

    Δy = heights_in[k .+ 1] .- h_k

    s = Δy ./ w
    ξ = (x_in .- w_k) ./ w

    # Derivatives at knot-points
    # Note that we have (K - 1) knot-points, not K

    d_k = ones(T, size(x))
    d_k[mask] = derivatives[:, k]

    d_k = (k == 0) ? one(T) : derivatives[k]
    d_kplus1 = (k == K - 1) ? one(T) : derivatives[k + 1]

    # Eq. (14)
    numerator = Δy * (s * ξ^2 + d_k * ξ * (1 - ξ))
    denominator = s + (d_kplus1 + d_k - 2s) * ξ * (1 - ξ)
    g = h_k + numerator / denominator

    return g
end