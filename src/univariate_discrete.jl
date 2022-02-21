using SpecialFunctions: digamma

function emstep(comp::Geometric, samples::AbstractVector{Int}, weights::AbstractWeights, zz::AbstractVector; kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z

    Geometric(1 / (1 + mx))
end

function emstep(::Poisson, samples::AbstractVector{Int}, weights::AbstractWeights, zz::AbstractVector; kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z

    Poisson(mx)
end

function emstep(::Bernoulli, samples::Union{AbstractVector{Int}, AbstractVector{Bool}}, weights::AbstractWeights, zz::AbstractVector; reg=1e-12, kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z

    Bernoulli(mx)
end

function emstep(comp::Binomial, samples::AbstractVector{Int}, weights::AbstractWeights, zz::AbstractVector; kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z

    Binomial(comp.n, mx / comp.n)
end

# This is implemented using the algorithm by Huang et al.
function emstep(comp::NegativeBinomial, samples::AbstractVector{Int}, weights::AbstractWeights, zz::AbstractVector;
                p_eps=1e-3, r_eps=1e-3, reg=1e-12, kwargs...)
    dig = digamma(comp.r + 1e-12)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mδ = comp.r * sum(i -> zz[i] * weights[i] * (digamma(comp.r + samples[i] + reg) - dig), 1:length(samples)) / Z
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z

    beta = 1 - 1 / (1 - comp.p + reg) - 1 / log(comp.p + reg)

    theta = beta * mδ / (mx - (1 - beta) * mδ)

    theta = clamp(theta, p_eps, 1 - p_eps)

    new_r = -mδ / log(theta)
    new_r = max(new_r, r_eps)

    new_p = theta

    NegativeBinomial(new_r, new_p)
end