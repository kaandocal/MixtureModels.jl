module MixtureModels

using Distributions, StatsBase

export fit_em, emstep

include("./univariate_discrete.jl")
include("./univariate_continuous.jl")
include("./fit_em.jl")

emstep(comp::Distribution, samples::AbstractVector; kwargs...) = emstep(comp, samples, uweights(length(samples)); kwargs...)
emstep(comp::Distribution, samples::AbstractVector, weights::AbstractWeights; kwargs...) = emstep(comp, samples, weights, uweights(length(weights)); kwargs...)

end