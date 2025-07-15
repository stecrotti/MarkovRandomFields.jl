"""
A Metropolis Hastings sampler
"""
struct MHSampler{T<:AbstractVector{<:Integer}} <: MRFSampler 
    order :: T
end 
function MHSampler(model::MarkovRandomField)
    order = collect(eachvariable(model))
    return MHSampler(order)
end

function sample_new_value(rng::AbstractRNG, model::MarkovRandomField, state::AbstractVector{<:Integer}, i::Integer)
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

function step(
        rng::AbstractRNG,
        model::MRFModel,
        sampler::MHSampler, 
        state::AbstractVector{<:Integer}; kw...)

    StatsBase.shuffle!(sampler.order)

    for i in sampler.order
        xi_current = state[i]
        xi_new = sample_new_value(rng, model.mrf, state, i)
        prob_current = prob_new = ULogarithmic(1)
        for a in neighbors(model.mrf.graph, v_vertex(i))
            ∂a = neighbors(model.mrf.graph, f_vertex(a))
            fa = model.mrf.factors[a]
            state[i] = xi_new
            prob_new *= weight(fa, @view state[∂a])
            state[i] = xi_current
            prob_current *= weight(fa, @view state[∂a])
        end
        bias_i = model.mrf.variable_biases[i]
        prob_new *= weight(bias_i, xi_new)
        prob_current *= weight(bias_i, xi_current)
        logprob_ratio = log(prob_new / prob_current)
        r = min(exp(logprob_ratio), 1)
        accept = rand(rng) < r
        accept && (state[i] = xi_new)
    end

    return state, state
end