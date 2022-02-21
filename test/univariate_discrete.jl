using Distributions, StatsBase
using MixtureModels

function test_emstep(::Type{Geometric}, N=1000)
    p = rand()
    samples = rand(Geometric(p), N)
    weights = uweights(N)
    
    samples_hist = unique(samples)
    weights_hist = FrequencyWeights([ count(x -> x == s, samples) for s in samples_hist ])
            
    pred = emstep(Geometric(), samples, weights, ones(N))
    pred_hist = emstep(Geometric(), samples_hist, weights_hist, ones(N))
    
    isapprox(pred.p, p, atol=0.05) && isapprox(pred_hist.p, p, atol=0.05)
end

function test_emstep(::Type{Poisson}, N=1000)
    λ = exp(randn())
    samples = rand(Poisson(λ), N)
    weights = uweights(N)
    
    samples_hist = unique(samples)
    weights_hist = FrequencyWeights([ count(x -> x == s, samples) for s in samples_hist ])
    
    pred = emstep(Poisson(), samples, weights, ones(N))
    pred_hist = emstep(Poisson(), samples_hist, weights_hist, ones(N))
    
    isapprox(pred.λ, λ, rtol=0.05) && isapprox(pred_hist.λ, λ, rtol=0.05)
end

function test_emstep(::Type{Bernoulli}, N=1000)
    N = 1000
    p = rand()
    samples = rand(Bernoulli(p), N)
    weights = uweights(N)
    
    samples_hist = unique(samples)
    weights_hist = FrequencyWeights([ count(x -> x == s, samples) for s in samples_hist ])
    
    pred = emstep(Bernoulli(), samples, weights, ones(N))
    pred_hist = emstep(Bernoulli(), samples_hist, weights_hist, ones(N))
    
    isapprox(pred.p, p, atol=0.05) && isapprox(pred.p, p, atol=0.05)
end

function test_emstep(::Type{Binomial}, n, N=1000)
    p = rand()
    samples = rand(Binomial(n, p), N)
    weights = uweights(N)
    
    samples_hist = unique(samples)
    weights_hist = FrequencyWeights([ count(x -> x == s, samples) for s in samples_hist ])
    
    pred = emstep(Binomial(n), samples, weights, ones(N))
    pred_hist = emstep(Binomial(n), samples_hist, weights_hist, ones(N))
    
    pred.n == n && isapprox(pred.p, p, atol=0.05) && pred_hist.n == n && isapprox(pred_hist.p, p, atol=0.05)
end

function test_emstep(::Type{NegativeBinomial}, N=10000, max_iter=100)
    r = exp(randn())
    p = rand()
    samples = rand(NegativeBinomial(r, p), N)
    weights = uweights(N)
    
    samples_hist = unique(samples)
    weights_hist = FrequencyWeights([ count(x -> x == s, samples) for s in samples_hist ])
    
    pred = pred_hist = NegativeBinomial(1, 1)
    for i in 1:max_iter
        pred = emstep(pred, samples, weights, ones(N))
        pred_hist = emstep(pred, samples_hist, weights_hist, ones(N))
    end
    
    isapprox(pred.r, r, atol=0.1) && isapprox(pred.p, p, atol=0.05) &&  isapprox(pred_hist.r, r, atol=0.1) && isapprox(pred_hist.p, p, atol=0.05)
end

N_rep = 100
p_success = 0.8

for T in [ Geometric, Poisson, Bernoulli ]
    @test count(test_emstep(T) for i in 1:N_rep) >= p_success * N_rep
end

@test count(test_emstep(Binomial, 5) for i in 1:N_rep) >= p_success * N_rep
@test count(test_emstep(Binomial, 20) for i in 1:N_rep) >= p_success * N_rep

@test count(test_emstep(NegativeBinomial) for i in 1:N_rep) >= 0.7 * N_rep
