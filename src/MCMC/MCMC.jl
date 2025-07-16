module MCMC

using MarkovRandomFields: MarkovRandomField, nstates, logweight, domain, eachvariable, nvariables
using Random: AbstractRNG, default_rng, seed!, randexp
import StatsBase, StatsBase.sample
using LogarithmicNumbers: ULogarithmic
using IndexedFactorGraphs: v_vertex, f_vertex, neighbors

include("core.jl")
include("metropolis.jl")
include("gibbs.jl")


export sample

export MHSampler
export GibbsSampler

export Serial, MultiThread

export state

end # end module