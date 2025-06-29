module MarkovRandomFields

using IndexedFactorGraphs: AbstractFactorGraph

include("factor.jl")
include("markov_random_field.jl")

include("Test/Test.jl")
    
end # end module
