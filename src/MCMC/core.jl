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

abstract type MRFSampler end

abstract type Parallelism end
struct Serial <: Parallelism end
struct MultiThread <: Parallelism end


state(::MarkovRandomField, x, args...; kwargs...) = copy(x)

function init_observables(model::MarkovRandomField, sampler::MRFSampler,
    initial_state, observables, nsamples)

    obs_vec = [[o(model, initial_state; sampler, it=1)]
        for o in observables]
    for o in obs_vec
        sizehint!(o, nsamples)
    end
    return obs_vec
end

""""
Save observables in a vector.
In case of vector-valued observables any copy of the vector must be made within the observable function
"""
function save_observables!(obs_vec, observables, model::MarkovRandomField, 
    sampler::MRFSampler, state, it)

    for (o, o_vec) in zip(observables, obs_vec)
        push!(o_vec, o(model, state; sampler, it))
    end
    return nothing
end

function merge_observable_vecs(obs_dicts)
    names = keys(first(obs_dicts))
    @assert all(keys(o)==names for o in obs_dicts)

    return Dict(
        name => [o[name] for o in obs_dicts]
        for name in names
    )
end

# Sample one chain
function StatsBase.sample(
    model::MarkovRandomField,
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
    model::MarkovRandomField,
    sampler::MRFSampler,
    nsamples::Integer;
    observables = [state],
    nwarmup = 0,
    initial_state = sample_from_variable_biases(rng, model),
    kw...,
)
    # check initial state
    length(initial_state) == nvariables(model) ||
        throw(ArgumentError("Initial state length must match number of variables, got $(length(initial_state)) and $(nvariables(model))"))

    state = initial_state
    for it in 1:nwarmup
        state = step(rng, model, sampler, state; kw...)
    end

    obs_vec = init_observables(model, sampler, state, observables, nsamples-nwarmup)    

    for it in 1:nsamples-nwarmup-1
        state = step(rng, model, sampler, state; kw...)
        save_observables!(obs_vec, observables, model, sampler, state, it)
    end

    # return samples
    return Dict(Symbol(f) => o for (f, o) in zip(observables, obs_vec))
end

# Sample in parallel
function StatsBase.sample(
    model::MarkovRandomField,
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
    model::MarkovRandomField,
    sampler::MRFSampler,
    parallel::MultiThread,
    nsamples::Integer,
    nchains::Integer;
    initial_state = [sample_from_variable_biases(rng, model) for _ in 1:nchains],
    observables = [state],
    kw...,
)
    # if there are less chains than threads, send each chain to one thread
    if nchains <= Threads.nthreads()
        return _sample_nochunks(rng, model, sampler, nsamples, nchains, observables;
            initial_state, kw...)
    # if there are more chains than threads, group chains and send each group to one thread, as in AbstractMCMC.jl
    else
        return _sample_chunks(rng, model, sampler, nsamples, nchains, observables;
            initial_state, kw...)
    end

end

function _sample_nochunks(
        rng::AbstractRNG,
    model::MarkovRandomField,
    sampler::MRFSampler,
    nsamples::Integer,
    nchains::Integer,
    observables;
    initial_state = [sample_from_variable_biases(rng, model) for _ in 1:nchains],
    kw...,
)
    threads = 1:nchains
    rngs = [copy(rng) for _ in threads]
    models = [deepcopy(model) for _ in threads]
    samplers = [deepcopy(sampler) for _ in threads]
    seeds = rand(rng, UInt, nchains)

    obs_dicts = Vector{Any}(undef, nchains)

    Threads.@threads for i in threads
        seed!(rngs[i], seeds)
        obs_dict = StatsBase.sample(
            rngs[i],
            models[i],
            samplers[i],
            nsamples;
            initial_state = initial_state[i],
            observables
        )
        obs_dicts[i] = obs_dict
    end

    return merge_observable_vecs(obs_dicts)
end

# copied from AbstractMCMC.jl
function _sample_chunks(
    rng::AbstractRNG,
    model::MarkovRandomField,
    sampler::MRFSampler,
    nsamples::Integer,
    nchains::Integer,
    observables;
    initial_state = [sample_from_variable_biases(rng, model) for _ in 1:nchains],
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

    obs_dicts = Vector{Any}(undef, nchains)

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
                    initial_state = initial_state[chainidx],
                    observables
                )

                obs_dicts[chainidx] = samples
            end
        end
    end

    return merge_observable_vecs(obs_dicts)
end