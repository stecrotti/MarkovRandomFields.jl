using MarkovRandomFields
using MarkovRandomFields.Test, MarkovRandomFields.MCMC, MarkovRandomFields.Models
using Test
using Aqua
using Statistics
import UniformIsingModels
using LinearAlgebra
using IndexedFactorGraphs, IndexedGraphs
using Random


@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(MarkovRandomFields; ambiguities = false)
    Aqua.test_all(MarkovRandomFields)
end

@testset "Basics" begin
    include("basics.jl")
end

@testset "MCMC" begin
    @testset "Metropolis" begin
        include("MCMC/metropolis.jl")
    end
    @testset "Gibbs" begin
        include("MCMC/gibbs.jl")
    end
end

@testset "Models" begin
    @testset "Ising" begin
        include("Models/ising.jl")
    end
    @testset "K-SAT" begin
        include("Models/ksat.jl")
    end
    @testset "Potts" begin
        include("Models/potts.jl")
    end
end



nothing
