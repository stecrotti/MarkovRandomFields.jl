module MCMC

using MarkovRandomFields: MarkovRandomField, nstates, logweight, domain, eachvariable, nvariables
using Random: AbstractRNG, default_rng, seed!, randexp
import StatsBase, StatsBase.sample
using LogarithmicNumbers: ULogarithmic
using IndexedFactorGraphs: v_vertex, f_vertex, neighbors

""" 
A subtype of AbstractMCMC.AbstractModel to work with discrete variables from Markov Random Fields
"""
struct MRFModel{T<:MarkovRandomField}
    mrf :: T
end

abstract type MRFSampler end


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
end
sample_noalloc(w) = sample_noalloc(default_rng(), w)


"""
Sample a random configuration from the variables' priors
"""
function sample_from_variable_biases(rng::AbstractRNG, model::MarkovRandomField)
    return map(eachvariable(model)) do i
        bias = model.variable_biases[i]
        states = domain(model, i)
        w = [logweight(bias, x) for x in states] 
        StatsBase.sample(rng, states, StatsBase.weights(exp.(w)))
    end
end

abstract type Parallelism end
struct Serial <: Parallelism end
struct MultiThread <: Parallelism end

# Sample one chain
function StatsBase.sample(
    model::MRFModel,
    sampler::MRFSampler,
    nsamples::Integer; 
    kw...
)
    return StatsBase.sample(
        default_rng(), model, sampler, nsamples; kw...
    )
end
function StatsBase.sample(
    rng::AbstractRNG,
    model::MRFModel,
    sampler::MRFSampler,
    nsamples::Integer;
    nwarmup = 0,
    initial_state = sample_from_variable_biases(rng, model.mrf),
    kw...,
)

    # check sizes
    # check initial state
    length(initial_state) == nvariables(model.mrf) ||
        throw(ArgumentError("Initial state length must match number of variables, got $(length(initial_state)) and $(nvariables(model.mrf))"))

    state = copy(initial_state)
    for it in 1:nwarmup
        sample, state = step(rng, model, sampler, state; kw...)
    end

    samples = [copy(state)]
    sizehint!(samples, nsamples - nwarmup)

    for it in 1:nsamples-nwarmup-1
        sample, state = step(rng, model, sampler, state; kw...)
        push!(samples, copy(sample))
    end

    return samples
end

# Sample in parallel
function StatsBase.sample(
    model::MRFModel,
    sampler::MRFSampler,
    parallel::Parallelism,
    nsamples::Integer,
    nchains::Integer; 
    kw...
)
    return StatsBase.sample(
        default_rng(), model, sampler, parallel, nsamples, nchains; kw...
    )
end

# Sample in parallel using multi-threading
function StatsBase.sample(
    rng::AbstractRNG,
    model::MRFModel,
    sampler::MRFSampler,
    parallel::MultiThread,
    nsamples::Integer,
    nchains::Integer;
    initial_state = [sample_from_variable_biases(rng, model.mrf) for _ in 1:nchains],
    kw...,
)
    # if there are less chains than threads, send each chain to one thread
    if nchains <= Threads.nthreads()
        return _sample_nochunks(rng, model, sampler, nsamples, nchains;
            initial_state, kw...)
    # if there are more chains than threads, group chains and send each group to one thread
    else
        return _sample_chunks(rng, model, sampler, nsamples, nchains;
            initial_state, kw...)
    end

end

function _sample_nochunks(
        rng::AbstractRNG,
    model::MRFModel,
    sampler::MRFSampler,
    nsamples::Integer,
    nchains::Integer;
    initial_state = [sample_from_variable_biases(rng, model.mrf) for _ in 1:nchains],
    kw...,
)
    threads = 1:nchains
    rngs = [copy(rng) for _ in threads]
    models = [deepcopy(model) for _ in threads]
    samplers = [deepcopy(sampler) for _ in threads]
    seeds = rand(rng, UInt, nchains)

    chains = Vector{Vector{Vector{eltype(eltype(initial_state))}}}(undef, nchains)

    Threads.@threads for i in threads
        seed!(rngs[i], seeds)
        samples = StatsBase.sample(
            rngs[i],
            models[i],
            samplers[i],
            nsamples;
            initial_state = initial_state[i]
        )
        chains[i] = samples
    end

    return chains
end

function _sample_chunks(
    rng::AbstractRNG,
    model::MRFModel,
    sampler::MRFSampler,
    nsamples::Integer,
    nchains::Integer;
    initial_state = [sample_from_variable_biases(rng, model.mrf) for _ in 1:nchains],
    kw...,
)
    # Copy the random number generator, model, and sample for each thread
    nchunks = min(nchains, Threads.nthreads())
    interval = 1:nchunks
    # `copy` instead of `deepcopy` for RNGs: https://github.com/JuliaLang/julia/issues/42899
    rngs = [copy(rng) for _ in interval]
    models = [deepcopy(model) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]

    # If nchains/nchunks = m with remainder n, then the first n chunks will
    # have m + 1 chains, and the rest will have m chains.
    m, n = divrem(nchains, nchunks)

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    chains = Vector{Vector{Vector{eltype(eltype(initial_state))}}}(undef, nchains)

    for (i, _rng, _model, _sampler) in zip(interval, rngs, models, samplers)
        # the first n chunks have m+1 chains while the others have m chains
        if i <= n
            chainidx_hi = i * (m + 1)
            nchains_chunk = m + 1
        else
            chainidx_hi = i * m + n # n * (m + 1) + (i - n) * m
            nchains_chunk = m
        end
        chainidx_lo = chainidx_hi - nchains_chunk + 1
        chainidxs = chainidx_lo:chainidx_hi

        @sync begin
            Threads.@spawn for chainidx in chainidxs
                # Seed the chunk-specific random number generator with the pre-made seed.
                seed!(_rng, seeds[chainidx])

                # Sample a chain and save it to the vector.
                samples = StatsBase.sample(
                    _rng,
                    _model,
                    _sampler,
                    nsamples;
                    initial_state = initial_state[chainidx]
                )

                chains[chainidx] = samples
            end
        end
    end

    return chains
end

include("metropolis.jl")
include("gibbs.jl")

export MRFModel
export sample

export MHSampler
export GibbsSampler

# export MCMCThreads, MCMCSerial, MCMCDistributed
export Serial, MultiThread

end # end module