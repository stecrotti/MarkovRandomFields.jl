module Models

using MarkovRandomFields: Factor, MarkovRandomField
import MarkovRandomFields
using IndexedGraphs: IndexedGraph, ne, nv, edges
using LinearAlgebra: Symmetric
using IndexedFactorGraphs: pairwise_interaction_graph, nvariables
using StatsBase: mean

include("ising.jl")
include("ksat.jl")
include("potts.jl")

export IsingCoupling, IsingField, IsingMRF, magnetization
export KSATClause
export PottsCoupling, PottsField, PottsMRF

end # end module