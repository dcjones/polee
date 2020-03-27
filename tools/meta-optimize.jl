
# Try to find reasonable ADAM paremeters by doing grid search
# Not really something users of polee should have to worry about.

import Polee
using HypothesisTests, Statistics

function main()
    gibbs_filename, annotations_filename, genome_filename, reads_filename, output_filename = ARGS

    output = open(output_filename, "a")
    gibbs_samples = Polee.read_gibbs_samples(gibbs_filename)
    num_samples, n = size(gibbs_samples)

    sample = Polee.RNASeqSample(
        annotations_filename,
        genome_filename,
        reads_filename,
        Set{String}(),
        Set{String}(),
        no_bias=true)

    approx = Polee.LogitSkewNormalPTTApprox(:cluster)
    approx_samples = Array{Float32}(undef, (num_samples, n))
    xs = Array{Float32}(undef, n)
    pvalues = Array{Float64}(undef, n)

    initial_learning_rate0 = Polee.ADAM_INITIAL_LEARNING_RATE
    learning_rate_decay0 = Polee.ADAM_LEARNING_RATE_DECAY

    # first round
    # initial_learning_rates = [1.0, 5e-1, 1e-1]
    # initial_learning_rates = [1.0]
    # learning_rate_decays = [1e-3, 1e-2, 2e-2, 3e-2]

    # second round
    # initial_learning_rates = [5e-2, 1e-2, 5e-3]
    # learning_rate_decays = [1e-3, 1e-2, 2e-2, 3e-2]

    rvs = [0.95, 0.9, 0.85]
    # rvs = [0.65, 0.7, 0.75]
    rms = [0.65, 0.7, 0.75]

    # third round

    # println(output, "initial_learning_rate,learning_rate_decay,median_pvalue")
    # for initial_learning_rate in initial_learning_rates
    #     for learning_rate_decay in learning_rate_decays
    #         Polee.set_adam_initial_learning_rate!(initial_learning_rate)
    #         Polee.set_adam_learning_rate_decay!(learning_rate_decay)
    #         @show (Polee.ADAM_INITIAL_LEARNING_RATE, Polee.ADAM_LEARNING_RATE_DECAY)

    # learning_rate_drops = [0.65, 0.7, 0.75]
    # learning_rate_drop_stepss = [15, 20, 25]

    # for learning_rate_drop in learning_rate_drops
    #     for learning_rate_drop_steps in learning_rate_drop_stepss
    #         Polee.set_adam_learning_rate_drop!(learning_rate_drop)
    #         Polee.set_adam_learning_rate_drop_steps!(learning_rate_drop_steps)
    #         @show (Polee.ADAM_LEARNING_RATE_DROP, Polee.ADAM_LEARNING_RATE_DROP_STEPS)
    for rv in rvs
        for rm in rms
            Polee.set_adam_rv!(rv)
            Polee.set_adam_rm!(rm)
            @show (Polee.ADAM_RV, Polee.ADAM_RM)

            params = Polee.approximate_likelihood(
                approx, sample, use_efflen_jacobian=false)

            als = Polee.ApproxLikelihoodSampler()
            t = Polee.PolyaTreeTransform(params["node_parent_idxs"], params["node_js"])
            Polee.set_transform!(
                als, t, params["mu"], exp.(params["omega"]), params["alpha"])

            for i in 1:num_samples
                Polee.rand!(als, xs)
                xs ./= sample.effective_lengths
                xs ./= sum(xs)
                approx_samples[i, :] = xs
            end

            for j in 1:n
                pvalues[j] = pvalue(SignedRankTest(
                    Vector{Float64}(gibbs_samples[:,j]),
                    Vector{Float64}(approx_samples[:,j])))
            end

            # println(output, initial_learning_rate, ",", learning_rate_decay, ",", median(pvalues))
            # println(output, learning_rate_drop, ",", learning_rate_drop_steps, ",", median(pvalues))
            println(output, rv, ",", rm, ",", median(pvalues))
            flush(output)
        end
    end
end


main()
