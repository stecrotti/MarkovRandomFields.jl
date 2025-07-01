using MarkovRandomFields
using MarkovRandomFields.Test, MarkovRandomFields.MCMC, MarkovRandomFields.Models
using Test
using Aqua
using Statistics
using AbstractMCMC
using UniformIsingModels: UniformIsing, energy, avg_energy
using LinearAlgebra
using IndexedFactorGraphs


# @testset "Code quality (Aqua.jl)" begin
#     Aqua.test_all(MarkovRandomFields; ambiguities = false)
#     Aqua.test_all(MarkovRandomFields)
# end

@testset "Basics" begin
    include("basics.jl")
end

@testset "MCMC" begin
    @testset "Metropolis" begin
        include("MCMC/metropolis.jl")
    end
end

@testset "Models" begin
    @testset "Ising" begin
        include("Models/ising.jl")
    end
end


nothing
