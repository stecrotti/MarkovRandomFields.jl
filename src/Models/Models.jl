module Models

using MarkovRandomFields: Factor, MarkovRandomField
import MarkovRandomFields
using IndexedGraphs: IndexedGraph, ne, nv
using LinearAlgebra: Symmetric
using IndexedFactorGraphs: pairwise_interaction_graph, nvariables
using StatsBase: mean

include("ising.jl")
include("ksat.jl")

export IsingCoupling, IsingField, IsingMRF, magnetization
export KSATClause

end # end module