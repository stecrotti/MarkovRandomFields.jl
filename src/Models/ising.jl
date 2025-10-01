potts2spin(x) = 3 - 2x

@doc raw"""
    IsingCoupling

A type of [`Factor`](@ref) representing a factor in an Ising distribution.
It involves $\pm 1$ variables $\boldsymbol{\sigma}_a=\{\sigma_1, \sigma_2, \ldots\}$.
The factor evaluates to
``\psi(\boldsymbol{\sigma}_a)=e^{\beta J \prod_{i\in a}\sigma_i}``.
A particular case is the pairwise interaction where $a=\{i,j\}$ is a pair of vertices involved in an edge $(ij)$.

Fields
========

- `βJ`: coupling strength.
"""
struct IsingCoupling{T<:Real}  <: Factor 
    βJ :: T 
end

function MarkovRandomFields.logweight(f::IsingCoupling{T}, x) where {T} 
    if isempty(x)
        return zero(T)
    else
        return f.βJ * prod(potts2spin(xᵢ) for xᵢ in x)
    end
end


@doc raw"""
    IsingField

A type of [`Factor`](@ref) representing a single-variable external field in an Ising distribution.
The factor evaluates to
``\psi(\sigma_i)=e^{\beta h \sigma_i}``.

Fields
========

- `βh`: field strength.
"""
struct IsingField{T<:Real}  <: Factor 
    βh :: T 
end

MarkovRandomFields.logweight(f::IsingField, x)  = f.βh * potts2spin(only(x))

const IsingMRF = MarkovRandomField{<:IsingCoupling, <:IsingField}

function IsingMRF(J::AbstractMatrix{<:Real}, h::Vector{<:Real}, β::Real=1)
    Jvec = [J[i,j] for j in axes(J,2) for i in axes(J,1) if i < j && J[i,j]!=0]
    g = IndexedGraph(Symmetric(J, :L))
    return IsingMRF(g, Jvec, h, β)
end

function IsingMRF(g::IndexedGraph, J::AbstractVector{<:Real}, 
    h::AbstractVector{<:Real}, β::Real=1)

    @assert length(J) == ne(g)
    @assert length(h) == nv(g)
    @assert β ≥ 0

    fg = pairwise_interaction_graph(g)
    factors = [IsingCoupling(β * Jᵢⱼ) for Jᵢⱼ in J]
    variable_biases = [IsingField(β * hᵢ) for hᵢ in h]
    nstates = fill(2, nvariables(fg))
    return MarkovRandomField(fg, factors, variable_biases, nstates)
end

## OBSERVABLES
magnetization(::IsingMRF, x, args...; kwargs...) = mean(potts2spin, x)
energy(model::IsingMRF, x, args...; kwargs...) = - logprob(model, x)