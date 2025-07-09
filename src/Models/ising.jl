
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

function MarkovRandomFields.weight(f::IsingCoupling, x) 
    if isempty(x)
        return 1
    else
        return exp(f.βJ * prod(potts2spin(xᵢ) for xᵢ in x))
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

MarkovRandomFields.weight(f::IsingField, x)  = exp(f.βh * potts2spin(only(x)))

# Ising model with xᵢ ∈ {1,2} mapped onto spins {+1,-1}
struct Ising{TJ<:Real, Th<:Real, Tβ<:Real, Tg<:Integer}
    g :: IndexedGraph{Tg}
    J :: Vector{TJ}
    h :: Vector{Th}
    β :: Tβ

    function Ising(g::IndexedGraph{Tg}, J::Vector{TJ}, h::Vector{Th}, β::Tβ=1) where 
            {TJ<:Real, Th<:Real, Tβ<:Real, Tg<:Integer}
        @assert length(J) == ne(g)
        @assert length(h) == nv(g)
        @assert β ≥ 0
        new{TJ, Th, Tβ, Tg}(g, J, h, β)
    end
end

function Ising(J::AbstractMatrix{<:Real}, h::Vector{<:Real}, β::Real=1.0)
    Jvec = [J[i,j] for j in axes(J,2) for i in axes(J,1) if i < j && J[i,j]!=0]
    g = IndexedGraph(Symmetric(J, :L))
    Ising(g, Jvec, h, β)
end

function MarkovRandomFields.MarkovRandomField(ising::Ising)
    g = pairwise_interaction_graph(ising.g)
    factors = [IsingCoupling(ising.β * Jᵢⱼ) for Jᵢⱼ in ising.J]
    variable_biases = [IsingField(ising.β * hᵢ) for hᵢ in ising.h]
    nstates = fill(2, nvariables(g))
    return MarkovRandomField(g, factors, variable_biases, nstates)
end