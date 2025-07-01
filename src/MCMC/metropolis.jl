"""
A Metropolis Hastings sampler
"""
struct MHSampler <: AbstractMCMC.AbstractSampler end 

function sample_new_value(rng::AbstractRNG, sampler::MHSampler, model::MarkovRandomField, state::AbstractVector{<:Integer}, i::Integer)
    xi = state[i]
    nstates_i = nstates(model, i)
    if nstates_i == 1
        error("TO BE DEALT WITH")
    end
    up = rand(rng, 1:(nstates_i - 1))
    xi_new = mod1(xi + up, nstates_i)
    @assert xi_new != xi
    return xi_new
end

###### AbstractMCMC interface: it is enough to override the `step` method in two versions: without and with the `state` argument, to deal with generation of the first and the subsequent samples, respectively
function AbstractMCMC.step(
        rng::AbstractRNG,
        model::MRFModel,
        sampler::MHSampler, 
        state::AbstractVector{<:Integer}; kw...)

    state_new = copy(state)
    i = rand(rng, eachindex(state))
    xi_new = sample_new_value(rng, sampler, model.mrf, state, i)
    state_new[i] = xi_new
    logprob_current = logprob(model.mrf, state)
    logprob_new = logprob(model.mrf, state_new)
    logprob_ratio = logprob_new - logprob_current
    if isnan(logprob_ratio)
        @show logprob_current, logprob_new
    end
    r = min(exp(logprob_ratio), 1)
    accept = rand(rng) < r
    accept && copy!(state, state_new)
    return copy(state), state
end

# First sample
function AbstractMCMC.step(
        rng::AbstractRNG,
        model::MRFModel,
        sampler::MHSampler;
        initial_state = Nothing, 
        kw...)

        if initial_state == Nothing
            next_state = sample_from_variable_biases(rng, model.mrf)
            return copy(next_state), next_state
        else
            return copy(initial_state), initial_state
        end
end
function AbstractMCMC.step(model::MRFModel, sampler::MHSampler, args...; kw...)
    return AbstractMCMC.step(default_rng(), model, sampler, args...; kw...)
end

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