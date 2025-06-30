using MarkovRandomFields, MarkovRandomFields.Test
using Test, TestItems
using Aqua

@testset "MarkovRandomFields.jl" begin
    # @testset "Code quality (Aqua.jl)" begin
    #     Aqua.test_all(MarkovRandomFields; ambiguities = false)
    #     Aqua.test_all(MarkovRandomFields)
    # end

    @testset "Basics" begin
        include("basics.jl")
    end
    
    @testset "Metropolis proposals" begin
        include("metropolis.jl")
    end
end

nothing
