using MarkovRandomFields, MarkovRandomFields.Test, MarkovRandomFields.MCMC
using Test
using Aqua
using Statistics
using AbstractMCMC

@testset "MarkovRandomFields.jl" begin
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
end

nothing
