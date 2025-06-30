@testset "Constructor" begin
    N = 2
    nstates = fill(3, N)
    A = zeros(0, 2)
    model = MarkovRandomField(A, TabulatedFactor[], nstates)
    @test all(isa.(model.variable_biases, UniformFactor))
end