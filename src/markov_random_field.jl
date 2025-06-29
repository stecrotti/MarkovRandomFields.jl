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

        new{TG, TF, TV, TS}(graph, factors, variable_biases)
    end
end
function MarkovRandomField(graph, factors, nstates; variable_biases=fill(UniformFactor, nvariables(graph)))
    return MarkovRandomField(graph, factors, variable_biases, nstates)
end

Base.isempty(model::MarkovRandomField) = isempty(model.graph)

nstates(model::MarkovRandomField, i::Integer) = model.nstates[i]

function energy(model::MarkovRandomField, x::AbstractVector{<:Integer})
    if isempty(model)
        return 0.0
    else
        return sum(energy(model.factors[a], x[∂a]) for a in factors(model.graph))
    end
end
function weight(model::MarkovRandomField, x::AbstractVector{<:Integer})
    return exp(-energy(model, x))
end
