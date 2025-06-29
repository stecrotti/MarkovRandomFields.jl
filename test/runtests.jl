using MarkovRandomFields
using Test
using Aqua

@testset "MarkovRandomFields.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MarkovRandomFields; ambiguities = false,)
    end
    # Write your tests here.
end
