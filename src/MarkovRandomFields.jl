module MarkovRandomFields

using IndexedFactorGraphs: AbstractFactorGraph, v_vertex, f_vertex, v_vertices, f_vertices, nvariables, nfactors, neighbors
import IndexedFactorGraphs
using Random: AbstractRNG, default_rng
using InvertedIndices: Not
using LogarithmicNumbers: ULogarithmic

using AbstractMCMC
using LogDensityProblems: LogDensityProblems, logdensity

include("factor.jl")
include("markov_random_field.jl")

include("metropolis.jl")

include("Test/Test.jl")


export TabulatedFactor, Factor

export MarkovRandomField
export nvariables, nfactors, domain, domains
export weight, logprob_unnormalized
export MRFLogDensityModel, MHSampler
    
end # end module
