module Models

using MarkovRandomFields: MarkovRandomFields, Factor

include("ising.jl")
include("ksat.jl")

export IsingCoupling, IsingField
export KSATClause

end # end module