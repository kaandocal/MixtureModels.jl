using Distributions, StatsBase
using MixtureModels

function test_emstep(::Type{Beta}, N=10000)
    α = rand()
    β = rand()
    samples = rand(Beta(α, β), N)
    weights = uweights(N)
            
    pred = emstep(Beta(), samples, weights, ones(N))
    
    isapprox(pred.α, α, atol=0.05) && isapprox(pred.β, β, atol=0.05)
end

function test_emstep(::Type{Normal}, N=10000)
    μ = randn()
    σ = exp(randn())
    samples = rand(Normal(μ, σ), N)
    weights = uweights(N)
            
    pred = emstep(Normal(), samples, weights, ones(N))
    
    isapprox(pred.μ, μ, atol=0.05) && isapprox(pred.σ, σ, rtol=0.05)
end

function test_emstep(::Type{Exponential}, N=1000)
    θ = exp(randn())
    samples = rand(Exponential(θ), N)
    weights = uweights(N)
    
    pred = emstep(Exponential(), samples, weights, ones(N))
    
    isapprox(pred.θ, θ, rtol=0.05)
end

function test_emstep(::Type{Gamma}, N=10000)
    α = exp(randn())
    θ = exp(randn())
    samples = rand(Gamma(α, θ), N)
    weights = uweights(N)
            
    pred = emstep(Gamma(), samples, weights, ones(N))
    
    isapprox(pred.α, α, atol=0.05) && isapprox(pred.θ, θ, rtol=0.05)
end

function test_emstep(::Type{Erlang}, α, N=1000)
    θ = exp(randn())
    samples = rand(Erlang(α, θ), N)
    weights = uweights(N)
            
    pred = emstep(Erlang(α), samples, weights, ones(N))
    
    pred.α == α && isapprox(pred.θ, θ, rtol=0.05)
end


N_rep = 100
p_success = 0.8

for T in [ Beta, Normal, Exponential, Gamma ]
    @test count(test_emstep(T) for i in 1:N_rep) >= p_success * N_rep
end

@test count(test_emstep(Erlang, 5) for i in 1:N_rep) >= p_success * N_rep
@test count(test_emstep(Erlang, 20) for i in 1:N_rep) >= p_success * N_rep
