function get_zz(components::AbstractVector, samples::AbstractVector)
    ret = [ pdf(comp, sample) for comp in components, sample in samples ]
    ret ./= sum(ret, dims=1)
end

function emstep(mixture::MT, samples::AbstractVector, weights::AbstractWeights; 
                zz_pre = get_zz(components(mixture), samples), 
                constructor = MixtureModel, kwargs...) where {MT <: AbstractMixtureModel}
    comps = components(mixture)
    
    comps_new = eltype(comps)[]
    for (i, comp) in enumerate(comps)
        zz_i = view(zz_pre, i, :)
        comp_new = emstep(comp, samples, weights, zz_i; kwargs...)
        push!(comps_new, comp_new)
    end
    
    p = get_zz(comps_new, samples) * weights
    p ./= sum(p)
    
    constructor(comps_new, p)
end

function fit_em(mixture::MT, samples::AbstractVector, weights::AbstractWeights; tol=1e-4, maxiter::Int=1000, kwargs...) where {MT <: AbstractMixtureModel}
    logl_hist = [ sum(i -> logpdf(mixture, samples[i]) * weights[i], 1:length(samples)) ]
    
    
    i = 1
    while true
        mixture = emstep(mixture, samples, weights; kwargs...)
        logl = sum(i -> logpdf(mixture, samples[i]) * weights[i], 1:length(samples))
        
        push!(logl_hist, logl)
        if abs(logl - logl_hist[end-1]) < tol
            break
        end
        
        i += 1
        if i > maxiter
            @warn "MixtureModels.em  failed to achieve convergence after $maxiter iterations"
            break
        end
    end
    
    mixture, logl_hist
end

fit_em(mixture::AbstractMixtureModel, samples::AbstractVector; kwargs...) = fit_em(mixture, samples, uweights(length(samples)); kwargs...)
emstep(mixture::AbstractMixtureModel, samples::AbstractVector; kwargs...) = emstep(mixture, samples, uweights(length(samples)); kwargs...)