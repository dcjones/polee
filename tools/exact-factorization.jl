#!/usr/bin/env julia

using ArgParse, HDF5, SparseArrays
import Polee

const arg_settings = ArgParseSettings()
arg_settings.prog = "approximate-factorization.jl"

@add_arg_table arg_settings begin
    "--output", "-o"
        default = "factorized-likelihood-matrix.h5"
        metavar = "factorized-likelihood-matrix.h5"
    "likelihood-matrix"
        metavar = "likelihood-matrix.h5"
        required = true
        help = """
            Likelihood matrix as generated with the 'prep' or 'prep-sample'
            commands using the '--likelihood-matrix' argument.
            """
end

function main()
    parsed_args = parse_args(arg_settings)
    sample = read(parsed_args["likelihood-matrix"], Polee.RNASeqSample)
    X = sample.X
    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))

    @show (m, n)

    # I guess we'll just hash every row?
    counts = Dict{Tuple{Vector{UInt32}, Vector{Float32}}, Int}()
    for i in 1:m
        if i % 100000 == 0
            println(i, " reads")
        end
        r = Xt.colptr[i]:Xt.colptr[i+1]-1
        key = (Xt.rowval[r], Xt.nzval[r])
        counts[key] = get(counts, key, 0) + 1
    end
    @assert sum(values(counts)) == m

    Is = UInt32[]
    Js = UInt32[]
    Vs = Float32[]
    Cs = UInt32[]
    frag_count = length(counts)

    for (i, ((js, vs), c)) in enumerate(counts)
        for (j, v) in zip(js, vs)
            push!(Is, i)
            push!(Js, j)
            push!(Vs, v)
        end
        push!(Cs, c)
    end
    @show frag_count

    X_compressed = sparse(Is, Js, Vs, frag_count, n)
    h5open(parsed_args["output"], "w") do output
        output["m"] = X_compressed.m
        output["n"] = X_compressed.n
        output["colptr", "compress", 1] = X_compressed.colptr
        output["rowval", "compress", 1] = X_compressed.rowval
        output["nzval", "compress", 1] = X_compressed.nzval
        output["counts", "compress", 1] = Cs
        output["effective_lengths", "compress", 1] = sample.effective_lengths
    end
end

main()

