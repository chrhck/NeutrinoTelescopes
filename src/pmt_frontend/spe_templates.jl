module SPETemplates

using Distributions

export SPEDistribution, ExponTruncNormalSPE
export make_spe_dist

"""
Abstract type for SPE distributions.

`Distribution` types can by created using [`make_spe_dist`]@ref	
"""
abstract type SPEDistribution{T<:Real} end

"""
Mixture model of an exponential and a truncated normal distribution
"""
struct ExponTruncNormalSPE{T<:Real}
    expon_rate::T
    norm_sigma::T
    norm_mu::T
    trunc_low::T
    expon_weight::T
end

"""
    make_spe_dist(d::SPEDistribution)

Return a `Distribution`
"""
make_spe_dist(d::SPEDistribution{T}) where {T} = error("not implemented")

function make_spe_dist(d::ExponTruncNormalSPE{T}) where {T<:Real}

    norm = Normal(d.norm_mu, d.norm_sigma)
    tnorm = truncated(norm, lower=d.trunc_low)

    expon = Exponential(d.expon_rate)
    dist = MixtureModel([expon, tnorm], [d.expon_weight, 1 - d.expon_weight])

    return dist
end
end