# MixtureModels.jl

Utilities for fitting mixture models to data extending [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). This is a new package; at the moment it implements the [Expectation-Maximisation algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) to fit mixture distributions to data using Maximum Likelihood Estimation (MLE). Suggestions and feedback very welcome.

## Usage:

Expectation-Maximisation is an iterative algorithm to fit mixture distributions to data. A single step of EM can be performed in closed form for many types of distributions including [exponential family distributions](https://en.wikipedia.org/wiki/Exponential_family). Every iteration of EM is guaranteed to increase the likelihood of the model, and a simple implementation of the fitting procedure stops when the likelihood stops changing meaningfully.

`fit_em(mixture::AbstractMixtureDistribution, samples::AbstractVector[, weights::AbstractWeights]; tol=1e-4, maxiter=1000, kwargs...)`

Fit a mixture to the given samples by iteratively performing EM steps until the total log likelihood improves by less than `tol`. The remaining `kwargs` are passed to `emstep`. Returns the fitted mixture and a list containing the total log likelihood after each iteration.

`emstep(mixture::AbstractMixtureDistribution, samples::AbstractVector[, weights::AbstractWeights]; constructor=MixtureModel)`

Perform an Expectation-Maximisation step for a mixture distribution. Here `constructor` is a function that takes a list of components and a weight vector and returns a new mixture.

`emstep(comp::Distribution, samples::AbstractVector[, weights::AbstractWeights]; kwargs...)`

Perform an Expectation-Maximisation step for a single distribution; in most cases this directly computes the MLE.

## Supported Distributions:

This package uses Julia's type dispatch and supports mixture distributions of various types (including mixtures of different distribution families).

### Univariate Discrete:
- `Geometric`
- `Poisson`
- `Bernoulli`
- `Binomial` (for fixed `n`)
- `NegativeBinomial` (following [#1])

### Univariate Continuous:
- `Beta`
- `Exponential`
- `Erlang` (for fixed `α`)
- `Gamma`
- `Normal`

## Related Packages:
- [MixFit.jl](https://github.com/the-sushi/MixFit.jl) supports a subset of the above distributions and uses moment matching instead of MLE for each EM step
- [MixtureModels.jl](https://github.com/lindahua/MixtureModels.jl) *(old)* is defunct
- [GaussianMixtures.jl](https://github.com/davidavdav/GaussianMixtures.jl) is specialised for Gaussian Mixture Models

## References

<a id="1">[1]</a> C. Huang, X. Liu, T. Yao and X. Wang, "An efficient EM algorithm for the mixture of negative binomial models", Journal of Physics Conference Series 1324(1), 044104 (2019). https://doi.org/10.1088/1742-6596/1324/1/012093
