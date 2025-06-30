module MCMC

using MarkovRandomFields: MarkovRandomField, nstates, logprob, weight, domain, variables
using AbstractMCMC
using Random: AbstractRNG, default_rng
using StatsBase: sample, weights

""" 
A subtype of AbstractMCMC.AbstractModel to work with discrete variables from Markov Random Fields
"""
struct MRFModel{T<:MarkovRandomField} <: AbstractMCMC.AbstractModel
    mrf :: T
end

include("metropolis.jl")

export MRFModel
export MHSampler

end # end module