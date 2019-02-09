
# Experiment with fitting gaussian graphical model to determine co-regulation of
# transcripts, splicing, gene expression, or whatever.

polee_dir = joinpath(dirname(@__FILE__), "..")

import Pkg
Pkg.activate(polee_dir)

using PyCall
pushfirst!(PyVector(pyimport("sys")["path"]), dirname(@__FILE__))
@pyimport coregulation as coregulation_py

import Polee
import YAML
using Random
using Statistics
using Profile

function main()
    # read specification
    spec = YAML.load_file(ARGS[1])
    if isempty(spec)
        error("Experiment specification is empty.")
    end

    if !haskey(spec, "samples")
        error("Experiment specification has no samples.")
    end

    excluded_transcripts = Set{String}()
    if length(ARGS) >= 2
        open(ARGS[2]) do input
            for line in eachline(input)
                push!(excluded_transcripts, chomp(line))
            end
        end
    end

    if haskey(spec, "annotations")
        transcripts_filename = spec["annotations"]
    else
        first_sample = first(spec["samples"])
        if haskey(first_sample, "file")
            first_sample_file = first_sample["file"]
        else
            if !haskey(first_sample, "name")
                error("Sample in experiment specification is missing a 'name' field.")
            end
            first_sample_file = string(first_sample["name"], prep_file_suffix)
        end

        transcripts_filename =
            Polee.read_transcripts_filename_from_prepared(first_sample_file)
        println("Using transcripts file: ", transcripts_filename)
    end

    Polee.init_python_modules()

    ts, ts_metadata = Polee.Transcripts(transcripts_filename, excluded_transcripts)
    n = length(ts)

    # Fit heirarchical model (of transcript expression)
    # TODO: this is very dangerous: we just load the file if it exists, even though
    # it may correspond to different data altogether.
    if isfile("qx_params.csv")
        # this is for testing, to save time
        open("qx_params.csv") do input
            row = split(readline(input), ',')
            sz1 = parse(Int, row[1])
            sz2 = parse(Int, row[2])
            qx_loc = Array{Float32}(undef, (sz1, sz2))
            qx_scale = Array{Float32}(undef, (sz1, sz2))
            for line in eachline(input)
                row = split(line, ',')
                i = parse(Int, row[1])
                j = parse(Int, row[2])
                qx_loc[i, j] = parse(Float32, row[3])
                qx_scale[i, j] = parse(Float32, row[4])
            end
        end
    else
        max_num_samples = nothing
        batch_size = nothing
        loaded_samples = Polee.load_samples_from_specification(
            spec, ts, ts_metadata, max_num_samples, batch_size)

        num_samples, n = size(loaded_samples.x0_values)
        x0_log = log.(loaded_samples.x0_values)
        qx_loc, qx_scale =
            Polee.polee_py[:estimate_transcript_expression](
                loaded_samples.init_feed_dict, num_samples, n,
                loaded_samples.variables, x0_log)
        Polee.tf[:reset_default_graph]()

        open("qx_params.csv", "w") do output
            println(output, size(qx_loc, 1), ",", size(qx_loc, 2))
            for i in 1:size(qx_loc, 1), j in 1:size(qx_loc, 2)
                println(output, i, ",", j, ",", qx_loc[i, j], ",", qx_scale[i, j])
            end
        end
    end

    expression_mode = exp.(qx_loc)
    expression_mode ./= sum(expression_mode, dims=2)
    idx = maximum(expression_mode, dims=1)[1,:] .> 1e-6
    idxmap = (1:n)[idx]
    @show sum(idx)

    qx_loc_subset = qx_loc[:,idx]
    qx_scale_subset = qx_scale[:,idx]

    # TODO: this will totally fuck up labels, but a useful test
    # p = shuffle(1:size(qx_loc_subset, 2))
    # qx_loc_subset = qx_loc_subset[:,p]
    # qx_scale_subset = qx_loc_subset[:,p]

    ts_names = [replace(t.metadata.name, "transcript:" => "") for t in ts]

    # Fit covariance matrix column by column
    @time edges = coregulation_py.estimate_gmm_precision(
        qx_loc_subset, qx_scale_subset)

    out = open("coregulation-graph.dot", "w")
    println(out, "graph coregulation {")
    for (u, vs) in edges
        i = ts_names[idxmap[u+1]]
        for (v, lower, upper) in vs
            j = ts_names[idxmap[v+1]]
            println(out, "    node", i, " -- node", j, ";")
        end
    end
    println(out, "}")
    close(out)

    out = open("coregulation-edges.csv", "w")
    for (u, vs) in edges
        i = ts_names[idxmap[u+1]]
        for (v, lower, upper) in vs
            j = ts_names[idxmap[v+1]]
            println(out, i, ",", j, ",", lower, ",", upper)
        end
    end
    close(out)

    # out = open("coregulation-expression.csv", "w")
    # for i in 1:n
    #     print(out, ts_names[i])
    #     for j in 1:size(qx_loc, 1)
    #         print(out, ",", qx_loc[j, i])
    #     end
    #     println(out)
    # end
    # close(out)

    degree = Dict{Int, Int}()
    for (u, vs) in edges
        if !haskey(degree, u)
            degree[u] = 0
        end
        degree[u] += length(vs)

        for (v, lower, upper) in vs
            if !haskey(degree, v)
                degree[v] = 0
            end
            degree[v] += 1
        end
    end

    out = open("coregulation-graph-degree.csv", "w")
    println(out, "transcript,degree")
    for (u, count) in degree
        println(out, ts_names[idxmap[u+1]], ",", count)
    end
    close(out)
end

main()
