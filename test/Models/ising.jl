@testset "Uniform Ising" begin
    N = 5
    J = 1.0
    h = zeros(N)

    A = ones(Int, N, N) - I
    graph = pairwise_interaction_graph(A)
    factors = fill(IsingCoupling(J/N), nfactors(graph))
    variable_biases = [IsingField(hi) for hi in h]
    nstates = fill(2, N)
    model = MarkovRandomField(graph, factors, variable_biases, nstates)
    ising = UniformIsing(N, J, h)

    x = rand(1:2, N)
    σ = MarkovRandomFields.Models.potts2spin.(x)
    @test -logprob(model, x) ≈ energy(ising, σ)

    avg_energy_exact = avg_energy(ising)

    # Run (in parallel!)
    nsamples = 10^5
    nchains = 8
    samples_bundle = sample(MRFModel(model), MHSampler(), MCMCThreads(),
        nsamples, nchains)
    samples = reduce(vcat, samples_bundle)
    energies_mcmc = [-logprob(model, x) for x in samples[end÷4:end]]
    avg_energy_mcmc = mean(energies_mcmc)
    @test abs(avg_energy_exact - avg_energy_mcmc) ≤ 1e-2
end