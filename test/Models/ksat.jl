@testset "ksat tree" begin
    A = [1 0 0 0 1; 0 1 0 0 0; 1 1 1 1 0]
    g = FactorGraph(A)
    factors = [KSATClause{BitVector}(Bool[0, 1]), KSATClause{BitVector}(Bool[1]), KSATClause{BitVector}(Bool[0, 0, 1, 1])]
    variable_biases = fill(TabulatedFactor([0.6, 0.4]), nvariables(g))
    model = MarkovRandomField(g, factors, fill(2, eachvariable(g)); variable_biases)
    p_ex = exact_prob(model)

    # test that probability of an unsat configuration is zero
    @test all(zip(eachstate(model), p_ex)) do (x, p)
        nunsat = sum(1 - Int(weight(model.factors[a], x[i] 
            for i in neighbors(model.graph, f_vertex(a)))) 
            for a in eachfactor(model.graph))
        (nunsat == 0) ⊻ (p == 0)
    end
end

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
        nunsat = sum(1 - Int(weight(model.factors[a], x[i] 
            for i in neighbors(model.graph, f_vertex(a)))) 
            for a in eachfactor(model.graph))
        (nunsat == 0) ⊻ (p == 0)
    end
end