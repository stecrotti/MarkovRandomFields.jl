# Custom proposal type for discrete MRF variables
struct MRFProposal{T} <: AdvancedMH.Proposal
    flip_prob::Float64  # Probability of flipping each variable
    rng::T
end

MRFProposal(flip_prob::Float64 = 0.1) = MRFProposal(flip_prob, Random.default_rng())

# Make it a proper proposal by implementing the required interface
Base.@kwdef struct DiscreteRandomWalkProposal{T} <: AdvancedMH.Proposal{T}
    proposal::T
end

# Custom sampler type for MRF models
struct MRFMH{P} <: AbstractMCMC.AbstractSampler
    proposal::P
end

# Wrapper for the MRF model to work with AdvancedMH
struct MRFDensityModel{T}
    mrf::T
end

# Implement LogDensityProblems interface for the MRF model
function LogDensityProblems.logdensity(model::MRFDensityModel, x)
    return logprobability_unnormalized(model.mrf, x)
end

function LogDensityProblems.dimension(model::MRFDensityModel)
    # Assuming the MRF model has a field that gives us the number of variables
    return IndexedFactorGraphs.nvariables(model.mrf)  # Adjust based on your MRF implementation
end

LogDensityProblems.capabilities(::Type{<:MRFDensityModel}) = 
    LogDensityProblems.LogDensityOrder{0}()

# Custom proposal function for MRF variables
function AdvancedMH.propose(rng::AbstractRNG, proposal::MRFProposal, x)
    # x is the current state (array of discrete variables)
    x_new = copy(x)
    
    # Randomly flip variables based on flip_prob
    for i in eachindex(x_new)
        if rand(rng) < proposal.flip_prob
            # For variables with arbitrary discrete domains, randomly select new value
            domain = get_variable_domain(proposal, i)  # Get domain for variable i
            # Exclude current value and pick randomly from remaining values
            current_val = x_new[i]
            other_vals = filter(v -> v != current_val, domain)
            if !isempty(other_vals)
                x_new[i] = rand(rng, other_vals)
            end
        end
    end
    
    return x_new
end

# Alternative: Single variable flip proposal (more standard for MRFs)
struct MRFSingleFlipProposal{T}
    mrf::MarkovRandomField  # Need reference to MRF for domain information
    rng::T
end

MRFSingleFlipProposal(mrf::MarkovRandomField) = MRFSingleFlipProposal(mrf, default_rng())

function AdvancedMH.propose(rng::AbstractRNG, proposal::MRFSingleFlipProposal, x)
    x_new = copy(x)
    # Randomly select one variable to flip
    idx = rand(rng, eachindex(x))
    
    # Get domain for this variable and select different value
    domain = get_variable_domain(proposal.mrf, idx)
    current_val = x_new[idx]
    other_vals = filter(v -> v != current_val, domain)
    
    if !isempty(other_vals)
        x_new[idx] = rand(rng, other_vals)
    end
    
    return x_new
end

# Gibbs-style proposal: sample from conditional distribution
struct MRFGibbsProposal{T}
    mrf::MarkovRandomField  # Need reference to MRF for conditional sampling
    rng::T
end

MRFGibbsProposal(mrf::MarkovRandomField) = MRFGibbsProposal(mrf, default_rng())

function AdvancedMH.propose(rng::AbstractRNG, proposal::MRFGibbsProposal, x)
    x_new = copy(x)
    # Randomly select one variable to update via Gibbs sampling
    idx = rand(rng, eachindex(x))
    
    # Sample from conditional distribution for this variable
    x_new[idx] = conditional_sample(proposal.mrf, x, idx, rng)
    
    return x_new
end

# For symmetric proposals, the log probability ratio is 0
function AdvancedMH.logratio_proposal_density(
    proposal::Union{MRFProposal, MRFSingleFlipProposal}, 
    x_new, 
    x_old
)
    return 0.0  # Symmetric proposal
end

# For Gibbs proposals, need to compute the actual ratio
function AdvancedMH.logratio_proposal_density(
    proposal::MRFGibbsProposal, 
    x_new, 
    x_old
)
    # For Gibbs sampling, this should account for the proposal probabilities
    # Implementation depends on your specific conditional distributions
    return gibbs_proposal_ratio(proposal.mrf, x_new, x_old)
end

# Convenience constructor
function mrf_sampler(mrf_model, proposal_type=:single_flip; flip_prob=0.1)
    if proposal_type == :single_flip
        prop = MRFSingleFlipProposal(mrf_model)
    elseif proposal_type == :gibbs
        prop = MRFGibbsProposal(mrf_model)
    else
        error("Unknown proposal type: $proposal_type. Use :single_flip, :multi_flip, or :gibbs")
    end
    
    density_model = MRFDensityModel(mrf_model)
    return density_model, AdvancedMH.MetropolisHastings(prop)
end

# Example usage function
function sample_mrf_model(mrf_model, n_samples::Int; 
                         initial_state=nothing,
                         proposal_type=:single_flip,
                         flip_prob=0.1)
    
    # Set up the model and sampler
    model, sampler = mrf_sampler(mrf_model, proposal_type; flip_prob=flip_prob)
    
    # Initial state (random if not provided)
    if initial_state === nothing
        # Generate random initial configuration based on MRF structure
        dim = LogDensityProblems.dimension(model)
        initial_state = generate_random_state(mrf_model, dim)
    end
    
    # Sample using AbstractMCMC interface
    chain = AbstractMCMC.sample(
        model, 
        sampler, 
        n_samples; 
        initial_params=initial_state,
        chain_type=Vector{Vector}  # Store as vector of variable configurations
    )
    
    return chain
end

# Alternative: Direct AdvancedMH interface (if you prefer)
function sample_mrf_advancedmh(mrf_model, n_samples::Int;
                              initial_state=nothing,
                              proposal_type=:single_flip,
                              flip_prob=0.1)
    
    # Wrap the log probability function
    logp(x) = logprobability_unnormalized(mrf_model, x)
    density_model = AdvancedMH.DensityModel(logp)
    
    # Choose proposal
    if proposal_type == :single_flip
        prop = MRFSingleFlipProposal(mrf_model)
    elseif proposal_type == :multi_flip
        prop = MRFMultiFlipProposal(mrf_model, flip_prob)
    elseif proposal_type == :gibbs
        prop = MRFGibbsProposal(mrf_model)
    else
        error("Unknown proposal type")
    end
    
    sampler = AdvancedMH.MetropolisHastings(prop)
    
    # Initial state
    if initial_state === nothing
        # You'll need to determine dimension from your MRF model
        dim = IndexedFactorGraphs.nvariables(mrf_model)  # Adjust this line
        initial_state = generate_random_state(mrf_model, dim)
    end

    # Sample
    chain = AdvancedMH.sample(
        density_model,
        sampler,
        n_samples;
        initial_params=initial_state,
        chain_type=Vector{Vector}
    )
    
    return chain
end

# Helper functions that you would need to implement based on your MRF structure
function get_variable_domain(mrf::MarkovRandomField, var_idx::Int)
    # Return the possible values for variable var_idx
    # This is a placeholder - implement based on your MRF structure
    return domain(mrf, var_idx)  # Assuming domains is a field in your MRF
end

function generate_random_state(mrf::MarkovRandomField, dim::Int)
    state = [rand(domain_i) for domain_i in domains(mrf)]
    return state
end

function conditional_sample(mrf::MarkovRandomField, x, var_idx::Int, rng::AbstractRNG)
    # Sample from the conditional distribution P(X_i | X_{-i})
    # This is a key function you'd need to implement for your specific MRF
    # It should return a sample from the conditional distribution
    
    # Placeholder implementation
    domain = get_variable_domain(mrf, var_idx)
    # Compute conditional probabilities and sample accordingly
    # This would involve computing the conditional distribution based on
    # the MRF's clique potentials and the current state of other variables
    return rand(rng, domain)  # Simplified - implement proper conditional sampling
end

function gibbs_proposal_ratio(mrf::MarkovRandomField, x_new, x_old)
    # Compute the proposal ratio for Gibbs sampling
    # For true Gibbs sampling, this is typically 1 (ratio = 0 in log space)
    # But if using approximate conditional sampling, you'd compute the actual ratio
    return 0.0  # Placeholder
end
