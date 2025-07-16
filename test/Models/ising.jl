@testset "Uniform Ising" begin
    N = 5
    J = 1.0
    h = ones(N)

    A = ones(Int, N, N) - I 
    model = IsingMRF(J/N * A, h)

    ising = UniformIsingModels.UniformIsing(N, J, h)

    x = rand(1:2, N)
    σ = MarkovRandomFields.Models.potts2spin.(x)
    @test -logweight(model, x) ≈ UniformIsingModels.energy(ising, σ)

    avg_energy_exact = UniformIsingModels.avg_energy(ising)

    # Run Metropolis-Hastings (in parallel!)
    nsamples = 10^5
    nchains = 8
    observables = [state, magnetization]
    obs = sample(model, MHSampler(model), MultiThread(),
        nsamples, nchains; nwarmup=10^3, observables)
    samples_bundle = obs[:state]
    samples = reduce(vcat, samples_bundle)
    magnetiz_bundle = obs[:magnetization]
    magnetiz = reduce(vcat, magnetiz_bundle)
    energies_mcmc = [-logweight(model, x) for x in samples[end÷4:end]]
    avg_energy_mcmc = mean(energies_mcmc)
    @test abs(avg_energy_exact - avg_energy_mcmc) ≤ 1e-2
    magnetiz_mcmc = mean(magnetiz)
    magnetiz_exact = mean(UniformIsingModels.site_magnetizations(ising))
    @test abs(magnetiz_mcmc - magnetiz_exact) ≤ 1e-2
end