LogDensityProblems.dimension(model::MarkovRandomField) = nvariables(model) 
LogDensityProblems.logdensity(model::MarkovRandomField, x) = logprob_unnormalized(model, x)

struct MRFLogDensityModel{T<:MarkovRandomField} <: AbstractMCMC.AbstractModel
    logdensity :: T
end

struct MHSampler <: AbstractMCMC.AbstractSampler end 

# function sample_variable(rng::AbstractRNG, sampler::MHSampler, state::AbstractVector{<:Integer})
#     i = rand(rng, eachindex(state))
#     return i
# end

function sample_new_value(rng::AbstractRNG, sampler::MHSampler, model::MarkovRandomField, state::AbstractVector{<:Integer}, i::Integer)
    xi = state[i]
    nstates_i = 1:nstates(model, i)
    if nstates_i == 1
        error("TO BE DEALT WITH")
    end
    state_new = rand(rng, nstates_i[Not(xi)])   
    return state_new
end

# function Base.rand(rng::AbstractRNG, sampler::MHSampler)
#     i = sample_variable(rng, sampler)
#     xi_new = sample_new_value(rng, sampler, i)
#     xnew[i] = xi_new
#     return xnew
# end

function AbstractMCMC.step(
        rng::AbstractRNG,
        model::MRFLogDensityModel,
        sampler::MHSampler, 
        state::AbstractVector{<:Integer})

    i = rand(rng, eachindex(state))
    xi_new = sample_new_value(rng, sampler, model.logdensity, state, i)
    xnew = copy(state)
    xnew[i] = xi_new
    x = state
    logprob_ratio = logdensity(model.logdensity, xnew) - logdensity(model.logdensity, x)
    r = min(exp(logprob_ratio), 1)
    accept = rand(rng) < r
    accept && copy!(x, xnew)
    return xnew, xnew
end

# First sample
function AbstractMCMC.step(
        rng::AbstractRNG,
        model::MRFLogDensityModel,
        sampler::MHSampler;
        initial_state = Nothing)

        if initial_state == Nothing
            next_state = [rand(Xi) for Xi in domains(model.logdensity)]
            return next_state, next_state
        else
            return state, state
        end
end
function AbstractMCMC.step(model::MRFLogDensityModel, sampler::MHSampler, args...; kw...)
    return AbstractMCMC.step(default_rng(), model, sampler, args...; kw...)
end
