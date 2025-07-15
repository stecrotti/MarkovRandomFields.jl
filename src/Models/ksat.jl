@doc raw"""
    KSATClause

A type of [`Factor`](@ref) representing a clause in a k-SAT formula.
It involves $\{0,1\}$ variables $\boldsymbol{x}_a=\{x_1, x_2, \ldots, x_k\}$.
The factor evaluates to
``\psi_a(\boldsymbol{x}_a)=1 - \prod_{i\in a}\delta(x_i, J^a_{i})``.

Fields
========

- `J`: a vector of booleans.
"""
struct KSATClause{T}  <: Factor where {T<:AbstractVector{<:Bool}}
    J :: T  # J = 1 if x appears negated, J = 0 otherwise (as in mezard montanari)
end

function MarkovRandomFields.logweight(f::KSATClause{T}, x) where {T} 
    isempty(x) && return zero(T)
    return any(xᵢ - 1 != Jₐᵢ for (xᵢ, Jₐᵢ) in zip(x, f.J)) |> log
end