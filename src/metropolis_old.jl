struct MCMCState{TM<:MarkovRandomField, TX<:AbstractVector{<:Integer}}
    model :: TM
    x :: TX
    xnew :: TX

    function MCMCState(model::TM, x::TX; xnew::TX=copy(x)) where {TM<:MarkovRandomField, TX<:AbstractVector{<:Integer}}
        @assert xnew !== x
        length(xnew) != length(x) && throw(ArgumentError("Lengths of `x` and `xnew` must correspond, got $(length(x)) and $(length(xnew))"))
        for (i, xi, Xi) in zip(eachindex(x), x, model.nstates)
            xi ≤ Xi  || throw(DomainError("Variable i=$i is defined to have at most $Xᵢ states, got value $xᵢ"))
        end
        return new{TM,TX}(model, x, xnew)
    end
end

function update!(mcmc_state::MCMCState)
    copy!(mcmc_state.x, mcmc_state.xnew)
    return nothing
end

struct RandomFlip{TS<:MCMCState}
    mcmc_state :: TS
end

function sample_variable(rng::AbstractRNG, p::RandomFlip)
    i = rand(rng, eachindex(p.mcmc_state.x))
    return i
end

function sample_new_value(rng::AbstractRNG, p::RandomFlip, i::Integer)
    (; model, x) = p.mcmc_state
    xi = x[i]
    nstates_i = 1:nstates(model, i)
    if nstates_i == 1
        error("TO BE DEALT WITH")
    end
    xi_new = rand(rng, nstates_i[Not(xi)])   
    return xi_new
end

function Base.rand(rng::AbstractRNG, p::RandomFlip)
    (; xnew) = p.mcmc_state
    i = sample_variable(rng, p)
    xi_new = sample_new_value(rng, p, i)
    xnew[i] = xi_new
    return xnew
end
