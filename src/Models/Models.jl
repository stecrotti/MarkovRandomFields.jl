module Models

using MarkovRandomFields: Factor, MarkovRandomField
import MarkovRandomFields
using IndexedGraphs: IndexedGraph, ne, nv
using LinearAlgebra: Symmetric
using IndexedFactorGraphs: pairwise_interaction_graph, nvariables

include("ising.jl")
include("ksat.jl")

export Ising, IsingCoupling, IsingField
export KSATClause

end # end module