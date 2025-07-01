module MCMC

using MarkovRandomFields: MarkovRandomField, nstates, logprob, weight, domain, eachvariable
using AbstractMCMC
using Random: AbstractRNG, default_rng
import StatsBase

""" 
A subtype of AbstractMCMC.AbstractModel to work with discrete variables from Markov Random Fields
"""
struct MRFModel{T<:MarkovRandomField} <: AbstractMCMC.AbstractModel
    mrf :: T
end

include("metropolis.jl")

export MRFModel
export MHSampler
export sample

end # end module