@doc raw"""
    PottsCoupling

A type of [`Factor`](@ref) representing a factor in a Potts distribution.
It involves `k` discrete variables $\boldsymbol{x}_a$, each taking a discrete value.
The factor evaluates to
``\psi(\boldsymbol{x}_a)=e^{\beta J(\boldsymbol{x}_a)}``.
A particular case is the pairwise interaction where $a=\{i,j\}$ is a pair of vertices involved in an edge $(ij)$.

Fields
========

- `βJ`: an Array with $k$ indices storing the (log) values of the factor for each input.
"""
struct PottsCoupling{T<:AbstractArray{<:Real}}  <: Factor 
    βJ :: T 
end

function MarkovRandomFields.logweight(f::PottsCoupling{T}, x) where {T} 
    if isempty(x)
        return zero(eltype(T))
    else
        return f.βJ[x...]
    end
end

@doc raw"""
    PottsField

A type of [`Factor`](@ref) representing a single-variable external field in a Potts distribution.
The factor evaluates to
``\psi(x_i)=e^{\beta h(x_i})``.

Fields
========

- `βh`: a vector storing the (log) values of the factor for each input.
"""
struct PottsField{T<:AbstractVector{<:Real}}  <: Factor 
    βh :: T 
end

MarkovRandomFields.logweight(f::PottsField, x)  = f.βh[only(x)]

const PottsMRF = MarkovRandomField{<:PottsCoupling, <:PottsField}

function PottsMRF(g::IndexedGraph, J::AbstractVector{<:AbstractArray}, 
    h::AbstractVector{<:AbstractVector}, β::Real=1)

    @assert length(J) == ne(g)
    @assert length(h) == nv(g)
    @assert β ≥ 0

    nstates = length.(h)
    # Check consistency between number of states
    for (i, j, ij) in edges(g)
        @assert size(J[ij]) == (nstates[i], nstates[j])
    end

    fg = pairwise_interaction_graph(g)
    factors = [PottsCoupling(β * Jᵢⱼ) for Jᵢⱼ in J]
    variable_biases = [PottsField(β * hᵢ) for hᵢ in h]
    return MarkovRandomField(fg, factors, variable_biases, nstates)
end

## OBSERVABLES
energy(model::PottsMRF, x, args...; kwargs...) = - logprob(model, x)