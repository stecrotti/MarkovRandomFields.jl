module Models

using MarkovRandomFields: MarkovRandomFields, Factor

include("ising.jl")

export IsingCoupling, IsingField

end # end module