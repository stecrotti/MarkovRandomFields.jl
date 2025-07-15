"""
A Gibbs sampler
"""
struct GibbsSampler{T<:AbstractVector{<:Integer}} <: MRFSampler 
    order :: T
end 
function GibbsSampler(model::MarkovRandomField)
    order = collect(eachvariable(model))
    return MHSampler(order)
end 

function step(
        rng::AbstractRNG,
        model::MRFModel,
        sampler::GibbsSampler, 
        state::AbstractVector{<:Integer}; kw...)

    StatsBase.shuffle!(sampler.order)

    for i in sampler.order
        ∂i = neighbors(model.mrf.graph, v_vertex(i))
        nstates_i = nstates(model.mrf, i)

        # conditional probability of xi given its neighbors
        pcond = Iterators.map(1:nstates_i) do xi
            state[i] = xi
            logp = logweight(model.mrf.variable_biases[i], (xi,))
            for a in ∂i
                ∂a = neighbors(model.mrf.graph, f_vertex(a))
                fa = model.mrf.factors[a]
                logp += logweight(fa, @inbounds @view state[∂a])
            end 
            exp(p)
        end
        state[i] = sample_noalloc(rng, pcond)
    end
    return state, state
end