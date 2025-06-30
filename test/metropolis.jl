A = [0 1; 1 0]
nstates = fill(3, 2)
model = rand_mrf(A, nstates)
x = ones(Int, 2)

@testitem "Random flip" begin
    mcmc_state = MCMCState(model, x)
    proposal = RandomFlip(mcmc_state)
    xnew = rand(proposal)
    @test all(xi ≤ Xi for (xi,Xi) in zip(xnew, model.nstates))
end

m = DensityModel(x -> weight(model, x))
mcmc_state = MCMCState(model, x)
p = MetropolisHastings(StaticProposal(RandomFlip(mcmc_state)))
c = sample(m, p, 100)