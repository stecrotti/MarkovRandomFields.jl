"""
A Metropolis Hastings sampler
"""
struct MHSampler <: MRFSampler end 

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
    prob_current = prob_new = ULogarithmic(1)
    for a in neighbors(model.mrf.graph, v_vertex(i))
        ∂a = neighbors(model.mrf.graph, f_vertex(a))
        fa = model.mrf.factors[a]
        prob_current *= weight(fa, state[∂a])
        prob_new *= weight(fa, state_new[∂a])
    end
    bias_i = model.mrf.variable_biases[i]
    prob_new *= weight(bias_i, xi_new)
    prob_current *= weight(bias_i, state[i])
    logprob_ratio = log(prob_new / prob_current)
    if isnan(logprob_ratio)
        @show logprob_current, logprob_new
    end
    r = min(exp(logprob_ratio), 1)
    accept = rand(rng) < r
    accept && copy!(state, state_new)
    return copy(state), state
end