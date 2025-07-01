module Test

using MarkovRandomFields: TabulatedFactor, MarkovRandomField, nstates, weight, eachvariable, eachfactor, logprob
using Random: AbstractRNG, default_rng
using LogarithmicNumbers
using InvertedIndices
using IndexedFactorGraphs

export exact_prob, exact_marginals, exact_factor_marginals
export rand_factor, rand_mrf


"""
    rand_factor([rng,], nstates)

Return a random `Factor` whose domain is specified by the iterable `nstates`.
"""
function rand_factor(rng::AbstractRNG, nstates)
    isempty(nstates) && return Factor(zeros(0))
    values = rand(rng, nstates...)
    return TabulatedFactor(values)
end
rand_factor(nstates) = rand_factor(default_rng(), nstates)

"""
    rand_mrf([rng], g::AbstractFactorGraph, nstates; variable_biases)

Return a `MarkovRandomField` with random factors.

`nstates` is an iterable containing the number of values that can be taken by each variable.
"""
function rand_mrf(rng::AbstractRNG, g::AbstractFactorGraph, nstates)
    factors = [rand_factor(rng, [nstates[i] for i in neighbors(g,f_vertex(a))]) for a in f_vertices(g)] 
    variable_biases = [rand_factor(rng, [nstates[i]]) for i in v_vertices(g)]
    return MarkovRandomField(g, factors, nstates; variable_biases)  
end
rand_mrf(g::AbstractFactorGraph, nstates) = rand_mrf(default_rng(), g, nstates)
rand_mrf(A::AbstractMatrix, args...; kw...) = rand_mrf(FactorGraph(A), args...; kw...)

function eachstate(model::MarkovRandomField)
    return Iterators.product((1:nstates(model, i) for i in eachvariable(model))...)
end

nstatestot(model::MarkovRandomField) = prod(nstates(model, i) for i in eachvariable(model); init=1)

"""
    exact_lognormalization(model::MarkovRandomField)

Exhaustively compute the natural logarithm of the normalization constant.
"""
function exact_lognormalization(model::MarkovRandomField)
    nstatestot(model) > 10^10 && @warn "Exhaustive computations on a system of this size can take quite a while."
    return log(sum(ULogarithmic(weight(model, x)) for x in eachstate(model)))
end

"""
    exact_prob(model::MarkovRandomField; logZ = exact_normalization(model))

Exhaustively compute the probability of each possible configuration of the variables.
"""
function exact_prob(model::MarkovRandomField; logZ = exact_lognormalization(model))
    p = [exp(logprob(model, x) - logZ) for x in eachstate(model)]
    return p
end

"""
    exact_marginals(model::MarkovRandomField; p_exact = exact_prob(model))

Exhaustively compute marginal distributions for each variable.
"""
function exact_marginals(model::MarkovRandomField; p_exact = exact_prob(model))
    dims = 1:ndims(p_exact)
    return map(eachvariable(model)) do i
        vec(sum(p_exact; dims=dims[Not(i)]))
    end
end

"""
    exact_factor_marginals(model::MarkovRandomField; p_exact = exact_prob(model))

Exhaustively compute marginal distributions for each factor.
"""
function exact_factor_marginals(model::MarkovRandomField; p_exact = exact_prob(model))
    dims = 1:ndims(p_exact)
    return map(eachfactor(model)) do a
        ∂a = neighbors(model, f_vertex(a))
        dropdims(sum(p_exact; dims=dims[Not(∂a)]); dims=tuple(dims[Not(∂a)]...))
    end
end

end # end module