
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
using SQLite


function read_splice_feature_names(num_features)
    db = SQLite.DB("genes.db")

    feature_names = Array{String}(undef, num_features)
    for row in SQLite.Query(db, "select * from splicing_features")
        feature_names[row.feature_num] =
            string(row.type, ":", row.seqname, ":",
                min(row.included_first, row.excluded_first),
                "-", max(row.included_last, row.excluded_last))
    end

    return feature_names
end



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
    gene_db = Polee.write_transcripts("genes.db", ts, ts_metadata)
    n = length(ts)

    num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
        Polee.gene_feature_matrix(ts, ts_metadata)

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

        open("qx_splice_params.csv") do input
            row = split(readline(input), ',')
            sz1 = parse(Int, row[1])
            sz2 = parse(Int, row[2])
            qx_splice_loc = Array{Float32}(undef, (sz1, sz2))
            qx_splice_scale = Array{Float32}(undef, (sz1, sz2))
            for line in eachline(input)
                row = split(line, ',')
                i = parse(Int, row[1])
                j = parse(Int, row[2])
                qx_splice_loc[i, j] = parse(Float32, row[3])
                qx_splice_scale[i, j] = parse(Float32, row[4])
            end
        end
    else
        max_num_samples = nothing
        batch_size = nothing
        loaded_samples = Polee.load_samples_from_specification(
            spec, ts, ts_metadata, max_num_samples, batch_size)

        num_samples, n = size(loaded_samples.x0_values)

        p = sortperm(collect(zip(gene_idxs, transcript_idxs)))
        gene_idxs = gene_idxs[p]
        transcript_idxs = transcript_idxs[p]

        sess = Polee.tf[:Session]()

        qx_loc, qx_scale = Polee.polee_py[:estimate_feature_expression](
            loaded_samples.init_feed_dict,
            loaded_samples.variables,
            num_samples, num_features, n,
            gene_idxs .- 1, transcript_idxs .- 1,
            sess)

        open("qx_params.csv", "w") do output
            println(output, size(qx_loc, 1), ",", size(qx_loc, 2))
            for i in 1:size(qx_loc, 1), j in 1:size(qx_loc, 2)
                println(output, i, ",", j, ",", qx_loc[i, j], ",", qx_scale[i, j])
            end
        end

        (num_features,
        splice_feature_idxs, splice_feature_transcript_idxs,
        splice_antifeature_idxs, splice_antifeature_transcript_idxs) =
            Polee.splicing_features(ts, ts_metadata, gene_db, alt_ends=false)

        splice_feature_indices = hcat(splice_feature_idxs .- 1, splice_feature_transcript_idxs .- 1)
        splice_antifeature_indices = hcat(splice_antifeature_idxs .- 1, splice_antifeature_transcript_idxs .- 1)

        qx_splice_loc, qx_splice_scale = Polee.polee_py[:estimate_splicing_log_ratios](
            loaded_samples.init_feed_dict,
            loaded_samples.variables,
            num_samples, num_features, n,
            splice_feature_indices, splice_antifeature_indices,
            sess)

        open("qx_splice_params.csv", "w") do output
            println(output, size(qx_splice_loc, 1), ",", size(qx_splice_loc, 2))
            for i in 1:size(qx_splice_loc, 1), j in 1:size(qx_splice_loc, 2)
               println(output, i, ",", j, ",", qx_splice_loc[i, j], ",", qx_splice_scale[i, j])
            end
        end

        Polee.tf[:reset_default_graph]()
    end

    exit()

    # splice_labels = [string("splice", i) for i in 1:size(qx_splice_loc, 2)]
    splice_labels = read_splice_feature_names(size(qx_splice_loc, 2))

    expression_mode = exp.(qx_loc)
    expression_mode ./= sum(expression_mode, dims=2)
    gene_idx = maximum(expression_mode, dims=1)[1,:] .> 1e-4

    qx_loc_subset = qx_loc[:,gene_idx]
    qx_scale_subset = qx_scale[:,gene_idx]

    splice_idx = minimum(qx_splice_scale, dims=1)[1,:] .< 0.1

    qx_splice_loc_subset = qx_splice_loc[:,splice_idx]
    qx_splice_scale_subset = qx_splice_scale[:,splice_idx]

    @show size(qx_loc_subset)
    @show size(qx_splice_loc_subset)

    # Merge gene expression and splicing features
    qx_merged_loc = hcat(qx_loc_subset, qx_splice_loc_subset)
    qx_merged_scale = hcat(qx_scale_subset, qx_splice_scale_subset)

    specific_labels = vcat(gene_ids[gene_idx], splice_labels[splice_idx])
    readable_labels = vcat(gene_names[gene_idx], splice_labels[splice_idx])

    # Shuffle so we can test on more interesting subsets
    idx = shuffle(1:size(qx_merged_loc, 2))
    qx_merged_loc = qx_merged_loc[:,idx]
    qx_merged_scale = qx_merged_scale[:,idx]
    specific_labels = specific_labels[idx]
    readable_labels = readable_labels[idx]

    # qx_merged_loc = qx_loc_subset
    # qx_merged_scale = qx_scale_subset

    # Fit covariance matrix column by column
    @time edges = coregulation_py.estimate_gmm_precision(
        qx_merged_loc, qx_merged_scale)

    out = open("coregulation-graph.dot", "w")
    println(out, "graph coregulation {")
    println(out, "    node [shape=plaintext];")

    used_node_ids = Set{Int}()
    for (u, vs) in edges
        for (v, lower, upper) in vs
            push!(used_node_ids, u)
            push!(used_node_ids, v)

            println("------")
            println((lower, upper))
            println(readable_labels[u], " -- ", readable_labels[v])
            # @show qx_merged_loc[:,u] .- mean(qx_merged_loc[:,u])
            @show qx_merged_loc[:,u]
            @show qx_merged_scale[:,u]
            # @show qx_merged_loc[:,v] .- mean(qx_merged_loc[:,v])
            @show qx_merged_loc[:,v]
            @show qx_merged_scale[:,v]
            println(cor(qx_merged_loc[:,u] .- mean(qx_merged_loc[:,u]),
                        qx_merged_loc[:,v] .- mean(qx_merged_loc[:,v])))
        end
    end
    for id in used_node_ids
        gene_name = readable_labels[id+1]
        println(out, "    node", id, " [label=\"", gene_name, "\"];")
    end

    for (u, vs) in edges
        for (v, lower, upper) in vs
            println(
                out, "    node", u, " -- node", v,
                " [color=", upper < 0.0 ? "darkgoldenrod2" : "dodgerblue4",
                ", penwidth=", 1 + 4*abs((upper+lower)/2),
                "];")
        end
    end
    println(out, "}")
    close(out)

    out = open("coregulation-edges.csv", "w")
    for (u, vs) in edges
        i = specific_labels[u+1]
        for (v, lower, upper) in vs
            j = specific_labels[v+1]
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
        println(out, specific_labels[u+1], ",", count)
    end
    close(out)
end

main()
