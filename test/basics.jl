@testset "Constructor" begin
    N = 5
    nstates = fill(3, N)
    A = rand(Bool, 3, N)
    model = rand_mrf(A, nstates)
    @test all(isa.(model.variable_biases, UniformFactor))
end