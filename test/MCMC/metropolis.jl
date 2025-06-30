function sample_mh(model::MarkovRandomField; nsamples=10^4)
    model_ = MRFModel(model)
    sampler = MHSampler()
    return sample(model_, sampler, nsamples)
end

function sample_mh_parallel(model::MarkovRandomField; nsamples=10^4)
    nchains = Base.Threads.nthreads()
    samples_bundle = sample(MRFModel(model), MHSampler(), MCMCThreads(), 
        nsamples, nchains)
    return reduce(vcat, samples_bundle)
end

function sample_mh_distributed(model::MarkovRandomField; nsamples=10^4)
    nchains = Base.Threads.nthreads()
    samples_bundle = sample(MRFModel(model), MHSampler(), MCMCDistributed(), 
        nsamples, nchains)
    return reduce(vcat, samples_bundle)
end

@testset "Uniform Factors" begin
    A = [1 1 1]
    nstates = fill(3, 3)
    model = MarkovRandomField(A, fill(UniformFactor(), 1), nstates)
    samples = sample_mh(model; nsamples=10^4)
    m = mean(samples[end÷2:end])
    @test all(x -> abs(x - 2) < 1e-1, m)
end

@testset "Only variable biases" begin
    A = [1 1 1 1 1]
    nstates = fill(2, 5)
    biases = [0, 0.5, 1, 0.9, 0.1]
    variable_biases = [TabulatedFactor([1-b, b]) for b in biases]
    model = MarkovRandomField(A, fill(UniformFactor(), 1), nstates;
        variable_biases)
    samples = sample_mh(model; nsamples=10^4)
    m = mean(samples[end÷2:end])
    @test all(abs(mi - 1 - bi) < 1e-1 for (mi, bi) in zip(m, biases))
end

@testset "Compare marginals" begin
    A = zeros(Int, 0, 3)
    nstates = fill(2, 3)
    biases = [0.2, 0.3, 0.8]
    variable_biases = [TabulatedFactor([1-b, b]) for b in biases]
    model = MarkovRandomField(A, TabulatedFactor[], nstates; variable_biases)
    samples = sample_mh(model; nsamples=10^4)
    m = mean(samples[end÷2:end])
    marg = exact_marginals(model)
    m_ex = [sum(eachindex(margi).*margi) for margi in marg]
    @test all(abs(mi - mi_ex) < 1e-1 for (mi, mi_ex) in zip(m, m_ex))
end

@testset "Parallel and distributed" begin
    A = [1 0 0 1;
         0 0 1 1;
         1 1 1 0
         0 1 0 1]
    nstates = fill(5, 4)
    model = rand_mrf(A, nstates)

    @testset "Parallel" begin
        samples = sample_mh_parallel(model; nsamples=10^4)
        m = mean(samples[end÷2:end])
        marg = exact_marginals(model)
        m_ex = [sum(eachindex(margi).*margi) for margi in marg]
        @test all(abs(mi - mi_ex) < 1e-1 for (mi, mi_ex) in zip(m, m_ex))
    end

    @testset "Distributed" begin
        samples = sample_mh_distributed(model; nsamples=10^4)
        m = mean(samples[end÷2:end])
        marg = exact_marginals(model)
        m_ex = [sum(eachindex(margi).*margi) for margi in marg]
        @test all(abs(mi - mi_ex) < 1e-1 for (mi, mi_ex) in zip(m, m_ex))
    end    

end

