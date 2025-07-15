@testset "ksat random graph" begin
    n = 10
    m = 3
    k = 3
    g = rand_regular_factor_graph(n, m, k)
    factors = [KSATClause(bitrand(degree(g, f_vertex(a)))) for a in eachfactor(g)]
    variable_biases = [TabulatedFactor(1.0 .+ 1e-4 * randn(2)) for _ in eachvariable(g)]
    model = MarkovRandomField(g, factors, fill(2, eachvariable(g)); variable_biases)
    p_ex = exact_prob(model)

    # test that probability of an unsat configuration is zero
    @test all(zip(eachstate(model), p_ex)) do (x, p)
        nunsat = sum(1 - exp(logweight(model.factors[a], x[i] 
            for i in neighbors(model.graph, f_vertex(a)))) 
            for a in eachfactor(model.graph))
        (nunsat == 0) ⊻ (p == 0)
    end

    @testset "Metropolis ksat" begin
        samples = sample_mh(model; nsamples=10^4)
        m = mean(samples[end÷4:end])
        marg = exact_marginals(model)
        m_ex = [sum(eachindex(margi).*margi) for margi in marg]
        @test all(abs.(m .- m_ex) .< 1e-1)
    end

    @testset "Gibbs ksat" begin
        samples = sample_gibbs(model; nsamples=10^6)
        m = mean(samples[end÷4:end])
        marg = exact_marginals(model)
        m_ex = [sum(eachindex(margi).*margi) for margi in marg]
        @test all(abs.(m .- m_ex) .< 1e-2)
    end
end

