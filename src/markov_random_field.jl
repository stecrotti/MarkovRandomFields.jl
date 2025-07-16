struct MarkovRandomField{TF<:Factor, TV<:Factor, TS<:Integer, TG<:AbstractFactorGraph}
    graph :: TG
    factors :: Vector{TF}
    variable_biases :: Vector{TV}
    nstates :: Vector{TS}

    function MarkovRandomField(graph::TG, factors::AbstractVector{TF}, variable_biases::AbstractVector{TV},
        nstates::AbstractVector{TS}) where {TG<:AbstractFactorGraph, TV<:Factor, TF<:Factor, TS<:Integer}
        
        nvar = nvariables(graph)
        nfact = nfactors(graph)
        length(factors) == nfact || throw(DimensionMismatch("Number of factor nodes in factor graph `graph`, $nfact, does not match length of `factors`, $(length(factors))"))
        length(variable_biases) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `graph`, $nvar, does not match length of `variable_biases`, $(length(variable_biases))"))
        length(nstates) == nvar || throw(DimensionMismatch("Number of variable nodes in factor graph `graph`, $nvar, does not match length of `nstates`, $(length(nstates))"))
        for i in eachindex(nstates)
            nstates[i] > 0 || throw(ArgumentError("Number of states for each variable must be greater than zero. Got $(nstates[i]) for variable $i"))
        end

        new{TF, TV, TS, TG}(graph, factors, variable_biases, nstates)
    end
end
function MarkovRandomField(graph, factors, nstates; variable_biases=fill(UniformFactor(), nvariables(graph)))
    return MarkovRandomField(graph, factors, variable_biases, nstates)
end
MarkovRandomField(A::AbstractMatrix, args...; kw...) = MarkovRandomField(FactorGraph(A), args...; kw...)

IndexedFactorGraphs.nvariables(model::MarkovRandomField) = nvariables(model.graph)
IndexedFactorGraphs.nfactors(model::MarkovRandomField) = nfactors(model.graph)

IndexedFactorGraphs.eachvariable(model::MarkovRandomField) = eachvariable(model.graph)
IndexedFactorGraphs.eachfactor(model::MarkovRandomField) = eachfactor(model.graph)

Base.isempty(model::MarkovRandomField) = isempty(eachvariable(model.graph))

nstates(model::MarkovRandomField, i::Integer) = model.nstates[i]
domain(model::MarkovRandomField, i::Integer) = 1:nstates(model, i)
domains(model::MarkovRandomField) = (1:Xi for Xi in model.nstates)

function logweight_factors(model::MarkovRandomField, x)
    lw = 0
    for a in eachfactor(model.graph)
        lw += logweight(model.factors[a], x[neighbors(model.graph, f_vertex(a))])
    end
    return lw
end

function logweight_variables(model::MarkovRandomField, x)
    lw = 0
    for i in eachvariable(model.graph)
        lw += logweight(model.variable_biases[i], x[i])
    end
    return lw
end

function logweight(model::MarkovRandomField, x)
    return logweight_factors(model, x) + logweight_variables(model, x)
end