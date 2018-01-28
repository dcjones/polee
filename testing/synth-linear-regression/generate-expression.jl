#!/usr/bin/env julia

using Distributions, BioSequences, GenomicFeatures, ProgressMeter, HATTries, SHA

include("../../src/transcripts.jl")


"""
Read a 0/1 design matrix from a text file.

Rows correspond to samples, columns to factors.
"""
function read_design_matrix(filename)
    rows = []
    for line in eachline(open(filename))
        row = map(s -> parse(Int, s), split(line))
        for x in row
            @assert x == 0 || x == 1
        end
        push!(rows, transpose(row))
    end
    num_samples = length(rows)

    bias_col = fill(1, (num_samples, 1))

    return hcat(bias_col, vcat(rows...))
end


function softmax(xs)
    exp_xs = exp.(xs)
    return exp_xs ./ sum(exp_xs)
end


function main()
    srand(1939139)

    design_filename, transcripts_filename = ARGS

    X = read_design_matrix(design_filename)
    ts, ts_metadata = Transcripts(transcripts_filename)

    num_samples, num_factors = size(X)
    n = length(ts)

    # these should mirror those in models/linear-regression.jl
    w_mu0 = 0.0
    w_sigma0 = 2.0
    w_bias_mu0 = log(1/n)
    w_bias_sigma0 = 10.0

    w = Array{Float64}((num_factors, n))
    w[1,:] = rand(MvNormal(fill(w_bias_mu0, n), fill(w_bias_sigma0, n)))
    for i in 2:num_factors
        w[i,:] = rand(MvNormal(fill(w_mu0, n), fill(w_sigma0, n)))
    end

    # error
    x_df = 5.0
    x_mu = X * w # num_samples by n
    x_sigma = exp.(rand(MvNormal(fill(-0.5, n), fill(1.0, n))))

    x = similar(x_mu)
    for i in 1:num_samples
        for j in 1:n
            # x[i, j] = rand(TDist(x_df, x_mu[i, j], x_sigma[i]))
            x[i, j] = x_mu[i, j] + rand(TDist(x_df)) * x_sigma[i]
            # x[i, j] = x_mu[i, j]
        end
    end

    for i in 1:num_samples
        x[i,:] = softmax(x[i,:])
    end

    open("transcript_ids.csv", "w") do out
        for t in ts
            println(out, t.metadata.name)
        end
    end

    open("w_true.csv", "w") do out
        println(out, join([string("factor", fctr) for fctr in 1:num_factors], ","))
        for j in 1:n
            println(out, join(w[:, j], ","))
        end
    end

    open("x_sigma_true.csv", "w") do out
        for j in 1:n
            println(out, x_sigma[j])
        end
    end

    open("x_true.csv", "w") do out
        println(out, join([string("sample", fctr) for fctr in 1:num_samples], ","))
        for j in 1:n
            println(out, join(x[:, j], ","))
        end
    end
end

main()

