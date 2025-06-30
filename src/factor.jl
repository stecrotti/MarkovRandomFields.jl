"""
    Factor

An abstract type representing a factor.
"""
abstract type Factor end

"""
    UniformFactor

A type of `Factor` which returns the same value for any input: it behaves as if it wasn't even there.
"""
struct UniformFactor <: Factor; end
weight(f::UniformFactor, x) = 1

"""
    TabulatedFactor

A type of `Factor` constructed by specifying the output to any input in a tabular fashion via an array `values`.
"""
struct TabulatedFactor{T<:Real,N} <: Factor
    values :: Array{T,N}
    function TabulatedFactor(values::Array{T,N}) where {T<:Real,N}
        any(<(0), values) && throw(ArgumentError("Factors can only take non-negative values"))
        return new{T,N}(values)
    end
end

function weight(f::TabulatedFactor, x) 
    isempty(x) && return one(eltype(f.values))
    return f.values[x...]
end