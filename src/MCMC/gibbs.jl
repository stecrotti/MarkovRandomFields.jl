"""
A Gibbs sampler
"""
struct GibbsSampler <: MRFSampler end 

function AbstractMCMC.step(
        rng::AbstractRNG,
        model::MRFModel,
        sampler::GibbsSampler, 
        state::AbstractVector{<:Integer}; kw...)

    state_new = copy(state)
    i = rand(rng, eachindex(state))
    ∂i = neighbors(model.mrf.graph, v_vertex(i))
    nstates_i = nstates(model.mrf, i)
    pcond = fill(ULogarithmic(1), nstates_i)
    for xi in 1:nstates_i
        state_new[i] = xi
        pcond[xi] *= weight(model.mrf.variable_biases[i], (xi,))
        for a in ∂i
            ∂a = neighbors(model.mrf.graph, f_vertex(a))
            fa = model.mrf.factors[a]
            pcond[xi] *= weight(fa, state_new[∂a])
        end 
    end
    state_new[i] = state[i] = StatsBase.sample(1:nstates_i, StatsBase.weights(pcond))
    return copy(state), state
end