
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