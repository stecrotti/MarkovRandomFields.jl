using AbstractMCMC
using MarkovRandomFields, MarkovRandomFields.Test
using Random

A = [0 1; 1 0]
nstates = fill(3, 2)
mrf_model = rand_mrf(A, nstates)
x = ones(Int, 2)

model = MRFLogDensityModel(mrf_model)
sampler = MHSampler()

a, b = AbstractMCMC.step(MersenneTwister(0), model, sampler)

itr = AbstractMCMC.steps(MersenneTwister(0), model, sampler)
samples = collect(Iterators.take(itr, 1000))