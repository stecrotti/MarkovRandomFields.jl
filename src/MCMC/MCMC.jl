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
        initial_state = sample_from_variable_biases(rng, model.mrf), 
        kw...)

        return copy(initial_state), initial_state
end
function AbstractMCMC.step(model::MRFModel, sampler::MRFSampler, args...; kw...)
    return AbstractMCMC.step(default_rng(), model, sampler, args...; kw...)
end

# SAMPLING
# sample an index `i` of `w` with probability prop to `w[i]`
# copied from StatsBase but avoids creating a `Weight` object
function sample_noalloc(rng::AbstractRNG, w) 
    t = rand(rng) * sum(w)
    i = 0
    cw = 0.0
    for p in w
        cw += p
        i += 1
        cw > t && return i
    end
    @assert false
end
sample_noalloc(w) = sample_noalloc(GLOBAL_RNG, w)


include("metropolis.jl")
include("gibbs.jl")

export MRFModel
export sample

export MHSampler
export GibbsSampler

export MCMCThreads, MCMCSerial, MCMCDistributed

end # end module