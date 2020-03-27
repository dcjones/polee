#!/usr/bin/env julia

# This is an implementation of Salmon style "range factorization" as described
#
# Zakeri, Mohsen, Avi Srivastava, Fatemeh Almodaresi, and Rob Patro. 2017.
# “Improved Data-Driven Likelihood Factorizations for Transcript Abundance
# Estimation.” Bioinformatics 33 (14): i142–51.
#
# This is just intended for benchmarking purposes. We want to measure the
# fidelity of this approximation, as well as the memory and cpu time necessariy
# to evaluate it.

using ArgParse, HDF5, SparseArrays
import Polee


const arg_settings = ArgParseSettings()
arg_settings.prog = "approximate-factorization.jl"

@add_arg_table arg_settings begin
    "--output", "-o"
        default = "factorized-likelihood-matrix.h5"
        metavar = "factorized-likelihood-matrix.h5"
    "--output-expanded"
        default = nothing
        metavar = "expanded-factorized-likelihood-matrix.h5"
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

    # group reads into  equivalence classes
    transcript_set_equiv_label = Dict{Vector{UInt32}, Int}()
    equiv_classes = Dict{Int, Vector{Int}}()

    for i in 1:m
        if i % 100000 == 0
            println(i, " reads")
        end
        r = Xt.colptr[i]:Xt.colptr[i+1]-1
        js = Xt.rowval[r]
        @assert !isempty(r)

        equiv_label = get!(
            transcript_set_equiv_label, js, 1+length(transcript_set_equiv_label))
        push!(get!(() -> Int[], equiv_classes, equiv_label), i)
    end

    println(length(transcript_set_equiv_label), " equivalence classes")

    Is = UInt32[]
    Js = UInt32[]
    Vs = Float32[]
    Cs = UInt32[]

    frag_count = 0
    for (equiv_class_label, equiv_class) in equiv_classes
        frag_count_q, Is_q, Js_q, Vs_q, Cs_q =
            approx_factorize_equiv_class(Xt, equiv_class)
        Is_q .+= frag_count
        frag_count += frag_count_q

        append!(Is, Is_q)
        append!(Js, Js_q)
        append!(Vs, Vs_q)
        append!(Cs, Cs_q)
    end

    println(length(Is), " entries")

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

    if parsed_args["output-expanded"] !== nothing
        Is_ex = UInt32[]
        Js_ex = UInt32[]
        Vs_ex = Float32[]

        p = sortperm(Is)
        Is = Is[p]
        Js = Js[p]
        Vs = Vs[p]

        frag_count = 0
        i_prev = 0
        for (i, j, v) in zip(Is, Js, Vs)
            if i != i_prev
                if i_prev > 0
                    frag_count += Cs[i_prev]
                end
                i_prev = i
            end

            for c in 1:Cs[i]
                push!(Is_ex, frag_count + c)
                push!(Js_ex, j)
                push!(Vs_ex, v)
            end
        end
        frag_count += Cs[last(Is)]

        X_expanded = sparse(Is_ex, Js_ex, Vs_ex, frag_count, n)
        h5open(parsed_args["output-expanded"], "w") do output
            output["m"] = X_expanded.m
            output["n"] = X_expanded.n
            output["colptr", "compress", 1] = X_expanded.colptr
            output["rowval", "compress", 1] = X_expanded.rowval
            output["nzval", "compress", 1] = X_expanded.nzval
            output["effective_lengths", "compress", 1] = sample.effective_lengths
        end
    end
end


function approx_factorize_equiv_class(Xt, equiv_class)
    r = Xt.colptr[first(equiv_class)]:Xt.colptr[first(equiv_class)+1]-1
    js = Xt.rowval[r]

    # this is the rule used in the paper
    num_bins = 4 + ceil(Int, sqrt(length(js)))

    # dense probability matrix for this equiv class
    xs = Array{Float64}(undef, (length(equiv_class), length(js)))

    for (u, i) in enumerate(equiv_class)
        xs[u, :] .= Xt.nzval[Xt.colptr[i]:Xt.colptr[i+1]-1]
    end

    # put probabilities in integer bins
    xs_min = minimum(xs, dims=1)
    xs_max = maximum(xs, dims=1)
    xs_span = xs_max .- xs_min
    xs_binsize = clamp.(xs_span ./ num_bins, 1f-16, Inf32)
    if !all(isfinite.((xs .- xs_min) ./ xs_binsize))
        @show xs
        @show xs_min
        @show xs .- xs_min
        @show xs_binsize

        exit()
    end
    xs_binned = floor.(Int, (xs .- xs_min) ./ xs_binsize)

    # count equivalence
    equiv_count = Dict{Vector{Int}, Int}()
    for i in 1:size(xs_binned, 1)
        xs_binned_i = xs_binned[i,:]
        equiv_count[xs_binned_i] = get!(equiv_count, xs_binned_i, 0) + 1
    end

    # expand into new entries
    num_equiv_frags = length(equiv_count)
    Is = Vector{UInt32}(undef, num_equiv_frags * length(js))
    Js = Vector{UInt32}(undef, num_equiv_frags * length(js))
    Vs = Vector{Float32}(undef, num_equiv_frags * length(js))
    Cs = Vector{UInt32}(undef, num_equiv_frags)

    Is = UInt32[]
    sizehint!(Is, num_equiv_frags * length(js))
    Js = UInt32[]
    sizehint!(Js, num_equiv_frags * length(js))
    Vs = Float32[]
    sizehint!(Vs, num_equiv_frags * length(js))
    Cs = UInt32[]
    sizehint!(Vs, num_equiv_frags)

    for (u, (xs_binned_i, count)) in enumerate(equiv_count)
        i = equiv_class[u]
        for (v, j) in enumerate(js)
            push!(Is, u)
            push!(Js, j)
            push!(Vs, xs_min[v] + (xs_binned_i[v] + 0.5) * xs_span[v])
        end
        push!(Cs, count)
    end

    return (length(equiv_count), Is, Js, Vs, Cs)
end


main()

