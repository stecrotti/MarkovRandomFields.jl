using MarkovRandomFields, MarkovRandomFields.Test
using Test, TestItems
using Aqua

@testset "MarkovRandomFields.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MarkovRandomFields; ambiguities = false,)
    end

    @testset "Basics" begin
        include("basics.jl")
    end
    
end

nothing
