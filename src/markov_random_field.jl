struct MarkovRandomField{TG<:AbstractFactorGraph, TF<:AbstractVector{<:Factor}, TV<:AbstractVector{<:Factor}, TS<:AbstractVector{<:Integer}}
    graph :: TG
    factors :: TF
    variable_biases :: TV
    nstates :: TS

    function MarkovRandomField(graph::TG, factors::TF, variable_biases::TV, nstates::TS) where {TG<:AbstractFactorGraph, TV<:AbstractVector{<:Factor}, TF<:AbstractVector{<:Factor}, TS<:AbstractVector{<:Integer}}
        nvar = nvariables(graph)
        nfact = nfactors(graph)
        length(factors) == nfact || throw(DimensionMismatch("Number of factor nodes in factor graph `graph`, $nfact, does not match length of `factors`, $(length(factors))"))
        length(variable_biases) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `graph`, $nvar, does not match length of `variable_biases`, $(length(variable_biases))"))
        length(nstates) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `graph`, $nvar, does not match length of `nstates`, $(length(nstates))"))
        for i in eachindex(nstates)
            nstates[i] > 0 || throw(ArgumentError("Number of states for each variable must be greater than zero. Got $(nstates[i]) for variable $i"))
        end

        new{TG, TF, TV, TS}(graph, factors, variable_biases, nstates)
    end
end
function MarkovRandomField(graph, factors, nstates; variable_biases=fill(UniformFactor(), IndexedFactorGraphs.nvariables(graph)))
    return MarkovRandomField(graph, factors, variable_biases, nstates)
end
MarkovRandomField(A::AbstractMatrix, args...; kw...) = MarkovRandomField(FactorGraph(A), args...; kw...)

IndexedFactorGraphs.nvariables(model::MarkovRandomField) = nvariables(model.graph)
IndexedFactorGraphs.nfactors(model::MarkovRandomField) = nfactors(model.graph)

Base.isempty(model::MarkovRandomField) = isempty(v_vertices(model.graph))

nstates(model::MarkovRandomField, i::Integer) = model.nstates[i]
domain(model::MarkovRandomField, i::Integer) = 1:nstates(model, i)
domains(model::MarkovRandomField) = (1:Xi for Xi in model.nstates)

function weight(model::MarkovRandomField, x::AbstractVector{<:Integer})
    p = one(ULogarithmic)
    for a in f_vertices(model.graph)
        p *= weight(model.factors[a], x[neighbors(model.graph, f_vertex(a))])
    end
    return p
end
logprob_unnormalized(model::MarkovRandomField, x) = log(weight(model, x))