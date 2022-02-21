using SafeTestsets

@time begin
    @safetestset "UnivariateDiscrete" begin include("univariate_discrete.jl") end
    @safetestset "UnivariateContinuous" begin include("univariate_continuous.jl") end
    @safetestset "Mixtures" begin include("mixtures.jl") end
end
