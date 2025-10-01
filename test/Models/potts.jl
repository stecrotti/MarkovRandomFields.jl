@testset "Reduce to Ising" begin
    potts2spin(x) = 3 - 2x

    N = 5
    q = fill(2, N)
    A = ones(Int, N, N) - I 
    g = IndexedGraph(A)
    J = randn(ne(g))
    h = zeros(N)
    JJ = [[J[ij] * potts2spin(xi) * potts2spin(xj) 
        for xi in 1:2, xj in 1:2] for (i,j,ij) in edges(g)]
    hh = [zeros(2) for i in 1:N]
    ising = IsingMRF(g, J, h)
    potts = PottsMRF(g, JJ, hh)

    @test all(Iterators.product(fill(1:2,N)...)) do x
        logweight(potts, x) ≈ logweight(ising, x)
    end
end