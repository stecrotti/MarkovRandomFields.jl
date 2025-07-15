"""
    Factor

An abstract type representing a factor.
It should implement the `weight(::Factor, x)` method to evaluate the factor for input `x`.
"""
abstract type Factor end

"""
    UniformFactor

A type of `Factor` which returns the same value for any input: it behaves as if it wasn't even there.
"""
struct UniformFactor <: Factor; end
logweight(f::UniformFactor, x) = 0

"""
    TabulatedFactor

A type of `Factor` constructed by specifying the output to any input in a tabular fashion via an array `values`.
"""
struct TabulatedFactor{T<:Real,N} <: Factor
    logweights :: Array{T,N}
    function TabulatedFactor(logweights::Array{T,N}) where {T<:Real,N}
        return new{T,N}(logweights)
    end
end

function logweight(f::TabulatedFactor, x) 
    isempty(x) && return zero(eltype(f.values))
    return f.logweights[x...]
end

weight(f::Factor, x) = exp(logweight(f, x))