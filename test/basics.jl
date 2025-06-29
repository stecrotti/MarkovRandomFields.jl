@testitem "Constructor" begin
    N = 5
    nstates = fill(3, N)
    model = rand_mrf(g, nstates)
    @test all(isa(UniformFactor), model.variable_biases)
end