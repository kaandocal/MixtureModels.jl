using Distributions, StatsBase
using MixtureModels

randinit(::Type{Beta}) = Beta(rand(), rand())
randinit(::Type{Gamma}) = Gamma(exp(randn()), exp(randn()))
randinit(::Type{Normal}) = Normal(randn(), exp(randn()))
randinit(::Type{Exponential}) = Exponential(exp(randn()))
randinit(::Type{Erlang}) = Erlang(rand(1:30), exp(randn()))

function test_single_cont(n, N=10000)
    types = [ Beta, Gamma, Normal, Exponential, Erlang ]
    comps = [ randinit(rand(types)) for i in 1:n ]
    
    p = rand(Dirichlet(ones(n) ./ n))
    target = MixtureModel(comps, p)
    
    samples = rand(target, N)
    
    comps_init = [ randinit(rand(types)) for i in 1:n ]
    mixture_init = MixtureModel(comps_init, ones(n) ./ n)
    
    mixture = fit_em(mixture_init, samples)
    
    isapprox(mean(mixture), mean(target), atol=0.05) && isapprox(std(mixture), std(target), atol=0.05)
end


randinit(::Type{Geometric}) = Geometric(rand())
randinit(::Type{Poisson}) = Poisson(exp(randn()))
randinit(::Type{Bernoulli}) = Bernoulli(rand())
randinit(::Type{Binomial}) = Binomial(rand(1:30), rand())
randinit(::Type{NegativeBinomial}) = NegativeBinomial(exp(randn()), rand())

function test_single_disc(n, N=10000)
    types = [ Geometric, Poisson, Bernoulli, Binomial, NegativeBinomial ]
    comps = [ randinit(rand(types)) for i in 1:n ]
    
    p = rand(Dirichlet(ones(n) ./ n))
    target = MixtureModel(comps, p)
    
    samples = rand(target, N)
    
    comps_init = [ randinit(rand(types)) for i in 1:n ]
    mixture_init = MixtureModel(comps_init, ones(n) ./ n)
    
    mixture = fit_em(mixture_init, samples)[1]
    
    isapprox(mean(mixture), mean(target), rtol=0.05) && isapprox(std(mixture), std(target), rtol=0.2)
end


N_rep = 20
p_success = 0.8

@test count(test_single_disc(rand(1:10)) for i in 1:N_rep) >= p_success * N_rep
