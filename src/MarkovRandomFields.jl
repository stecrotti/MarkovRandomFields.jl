module MarkovRandomFields

using IndexedFactorGraphs: AbstractFactorGraph, FactorGraph, v_vertex, f_vertex, v_vertices, f_vertices, nvariables, nfactors, neighbors
using IndexedFactorGraphs
using Random: AbstractRNG, default_rng
using InvertedIndices: Not
using LogarithmicNumbers: ULogarithmic

include("factor.jl")
include("markov_random_field.jl")

include("Test/Test.jl")
include("Models/Models.jl")
include("MCMC/MCMC.jl")


export Factor, TabulatedFactor, UniformFactor

export MarkovRandomField
export eachvariable, eachfactor, nvariables, nfactors, domain, domains
export weight, logprob

    
end # end module
