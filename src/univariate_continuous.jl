using LinearAlgebra: dot
using Distributions: gamma_mle_update
using SpecialFunctions: digamma, trigamma

function mod_fit(::Type{<:Beta}, x_bar, v_bar)
    temp = ((x_bar * (one(x_bar) - x_bar)) / v_bar) - one(x_bar)
    α = x_bar * temp
    β = (one(x_bar) - x_bar) * temp
    return Beta(α, β)
end

function mod_fit_mle(::Beta, g₁, g₂, x_bar, v_bar;
                    maxiter::Int=1000, tol::Float64=1e-12)
    α, β = params(mod_fit(Beta, x_bar, v_bar))
    θ= [α ; β ]

    converged = false
    t=0
    while !converged && t < maxiter #newton method
        t+=1
        temp1 = digamma(θ[1]+θ[2])
        temp2 = trigamma(θ[1]+θ[2])
        grad = [g₁+temp1-digamma(θ[1])
               temp1+g₂-digamma(θ[2])]
        hess = [temp2-trigamma(θ[1]) temp2
                temp2 temp2-trigamma(θ[2])]
        Δθ = hess\grad #newton step
        θ .-= Δθ
        converged = dot(Δθ,Δθ) < 2*tol #stopping criterion
    end

    return Beta(θ[1], θ[2])
end

function mod_fit_mle(start::Gamma, mx, mlogx;
    alpha0::Float64=NaN, maxiter::Int=1000, tol::Float64=1e-16)

    logmx = log(mx)

    a::Float64 = isnan(alpha0) ? (logmx - mlogx)/2 : alpha0
    converged = false

    t = 0
    while !converged && t < maxiter
        t += 1
        a_old = a
        a = gamma_mle_update(logmx, mlogx, a)
        converged = abs(a - a_old) <= tol
    end

    Gamma(a, mx / a)
end

safelog1mx(x, reg=1e-12) = log(clamp(one(x) - x, reg, 1 - reg))

function emstep(comp::Beta, samples::AbstractVector, weights::AbstractWeights, zz::AbstractVector; reg=1e-12, kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mlogx = sum(i -> zz[i] * weights[i] * log(samples[i] + reg), 1:length(samples)) / Z
    mlog1mx = sum(i -> zz[i] * weights[i] * safelog1mx(samples[i], reg), 1:length(samples)) / Z
    
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z
    mx2 = sum(i -> zz[i] * weights[i] * samples[i] ^ 2, 1:length(samples)) / Z

    mod_fit_mle(comp, mlogx, mlog1mx, mx, mx2 - mx^2; kwargs...)
end


function emstep(comp::Normal, samples::AbstractVector, weights::AbstractWeights, zz::AbstractVector; kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z
    mx2 = sum(i -> zz[i] * weights[i] * samples[i] ^ 2, 1:length(samples)) / Z
    
    Normal(mx, sqrt(mx2 - mx^2))
end

function emstep(::Exponential, samples::AbstractVector, weights::AbstractWeights, zz::AbstractVector; kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z

    Exponential(mx)
end

function emstep(comp::Gamma, samples::AbstractVector, weights::AbstractWeights, zz::AbstractVector; reg=1e-12, kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z
    mlogx = sum(i -> zz[i] * weights[i] * log(samples[i] + reg), 1:length(samples)) / Z

    mod_fit_mle(comp, mx, mlogx)
end

function emstep(comp::Erlang, samples::AbstractVector, weights::AbstractWeights, zz::AbstractVector; kwargs...)
    Z = sum(i -> zz[i] * weights[i], 1:length(samples))
    mx = sum(i -> zz[i] * weights[i] * samples[i], 1:length(samples)) / Z

    Erlang(comp.α, mx / comp.α)
end
