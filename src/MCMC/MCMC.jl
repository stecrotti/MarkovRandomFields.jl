module MCMC

using MarkovRandomFields: MarkovRandomField, nstates, logprob, weight, domain, eachvariable
using AbstractMCMC
using Random: AbstractRNG, default_rng
import StatsBase
using LogarithmicNumbers: ULogarithmic
using IndexedFactorGraphs: v_vertex, f_vertex, neighbors

""" 
A subtype of AbstractMCMC.AbstractModel to work with discrete variables from Markov Random Fields
"""
struct MRFModel{T<:MarkovRandomField} <: AbstractMCMC.AbstractModel
    mrf :: T
end

abstract type MRFSampler <: AbstractMCMC.AbstractSampler end

"""
Sample a random configuration from the variables' priors
"""
function sample_from_variable_biases(rng::AbstractRNG, model::MarkovRandomField)
    return map(eachvariable(model)) do i
        bias = model.variable_biases[i]
        states = domain(model, i)
        w = [weight(bias, x) for x in states] 
        StatsBase.sample(rng, states, StatsBase.weights(w))
    end
end

# First sample
function AbstractMCMC.step(
        rng::AbstractRNG,
        model::MRFModel,
        sampler::MRFSampler;
        initial_state = Nothing, 
        kw...)

        if initial_state == Nothing
            next_state = sample_from_variable_biases(rng, model.mrf)
            return copy(next_state), next_state
        else
            return copy(initial_state), initial_state
        end
end
function AbstractMCMC.step(model::MRFModel, sampler::MRFSampler, args...; kw...)
    return AbstractMCMC.step(default_rng(), model, sampler, args...; kw...)
end

include("metropolis.jl")
include("gibbs.jl")

export MRFModel
export sample

export MHSampler
export GibbsSampler

export MCMCThreads, MCMCSerial, MCMCDistributed

end # end module