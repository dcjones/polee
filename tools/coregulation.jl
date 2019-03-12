
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
using StatsBase
using InteractiveUtils
using SIMD

include("graphical-horseshoe.jl")


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


function randn!(loc, scale, col, out)
    for i in 1:size(loc, 1)
        out[i] = loc[i, col] + randn(Float32) * scale[i, col]
    end
end


function randn!(loc, scale, col, out_row, out)
    for i in 1:size(loc, 1)
        out[out_row, i] = loc[i, col] + randn(Float32) * scale[i, col]
    end
end


function connected_components(edges, n)
    visited = fill(false, n)
    num_visited = 0
    next_unvisited = 1
    components = Vector{Int}[]

    while num_visited < n
        while visited[next_unvisited]
            next_unvisited += 1
        end

        if !haskey(edges, next_unvisited)
            visited[next_unvisited] = true
            num_visited += 1
            continue
        end

        stack = Int[next_unvisited]
        component = Int[]

        while !isempty(stack)
            i = pop!(stack)
            if visited[i]
                continue
            end

            @assert !visited[i]
            @assert i ∉ component
            push!(component, i)

            visited[i] = true
            num_visited += 1

            for j in edges[i]
                push!(stack, j)
            end
        end

        push!(components, component)
    end

    return components
end


"""
We want to reduce the number of features we need to consider so we can reasonably
attempt to estimate a full precision matrix.

The heuristic we implement here is to exclude features that aren't
siginificantly correlated with another feature.
"""
function find_components(qx_loc, qx_scale)
    @assert size(qx_loc) == size(qx_scale)
    num_samples, n = size(qx_loc)

    @show n

    c = 0.75
    pr = 0.9
    num_reps = 20


    # how many repititions fall below the threshold c before an edge is rejected
    max_miss_reps = num_reps - floor(pr * num_reps)

    # standard normal rand numbers
    xs_std = Array{Float32}(undef, (num_samples, num_reps))
    ys_std = Array{Float32}(undef, (num_samples, num_reps))

    for u in 1:num_samples, v in 1:num_reps
        xs_std[u, v] = randn(Float32)
        ys_std[u, v] = randn(Float32)
    end

    # destandardized normals
    xss = [Array{Float32}(undef, (num_samples, num_reps)) for _ in 1:Threads.nthreads()]
    yss = [Array{Float32}(undef, num_samples) for _ in 1:Threads.nthreads()]

    # true if that index should be included (passes filter)
    edges = Dict{Int,Vector{Int}}()

    mut = Threads.Mutex()

    @time filter_features_inner(
        xss, yss, mut, edges, num_samples, num_reps, n, c, pr, xs_std, ys_std, qx_loc, qx_scale)

    @show div(sum([length(v) for (k, v) in edges]), 2)

    components = connected_components(edges, n)

    @show length(components)
    @show maximum(map(length, components))
    # @show collect(map(length, components))

    return components
end


function filter_features_inner(
        xss, yss, mut, edges, num_samples, num_reps, n,
        c, max_miss_reps, xs_std, ys_std, qx_loc, qx_scale)

    qx_loc_flat = reshape(qx_loc, (num_samples * n))
    qx_scale_flat = reshape(qx_scale, (num_samples * n))
    ys_std_flat = reshape(ys_std, (num_samples * num_reps,))

    Threads.@threads for i in 1:n
        thrid = Threads.threadid()
        xs = xss[thrid]
        ys = yss[thrid]

        xs_flat = reshape(xs, (num_samples * num_reps,))

        # destandardize x
        for u in 1:num_samples, v in 1:num_reps
            xs[u, v] = xs_std[u, v] * qx_scale[u, i] + qx_loc[u, i]
        end

        for j in i+1:n

            # TODO: only destardardize rows as needed, and use SIMD

            # destandardize y
            # for u in 1:num_samples, v in 1:num_reps
            #     ys[u, v] = ys_std[u, v] * qx_scale[u, j] + qx_loc[u, j]
            # end

            passed = filter_features_i(
                j, c, xs, xs_flat, ys, ys_std_flat,
                qx_loc_flat, qx_scale_flat, num_samples, num_reps, max_miss_reps)

            if passed
                lock(mut)
                if !haskey(edges, i)
                    edges[i] = Int[j]
                else
                    push!(edges[i], j)
                end

                if !haskey(edges, j)
                    edges[j] = Int[i]
                else
                    push!(edges[j], i)
                end
                unlock(mut)
            end
        end

        lock(mut)
        if haskey(edges, i)
            println(i, ": ", length(edges[i]), " edges found")
        end
        unlock(mut)
    end
end



function filter_features_i(
        j, c, xs, xs_flat, ys, ys_std_flat,
        qx_loc_flat, qx_scale_flat,
        num_samples, num_reps, max_miss_reps)

    # SIMD version
    num_samples_aligned = 8*div(num_samples, 8)

    miss_count = 0

    # @inbounds for t in 1:num_reps
    for t in 1:num_reps
        coloff = num_samples*(t-1)

        # destandardize ys
        for u in 1:8:num_samples_aligned
            z = vload(Vec{8,Float32}, ys_std_flat, coloff+u)
            loc = vload(Vec{8,Float32}, qx_loc_flat, num_samples*(j-1)+u)
            scale = vload(Vec{8,Float32}, qx_scale_flat, num_samples*(j-1)+u)
            vstore(z*scale+loc, ys, u)
        end

        for u in num_samples_aligned+1:num_samples
            z = ys_std_flat[coloff+u]
            loc = qx_loc_flat[num_samples*(j-1)+u]
            scale = qx_scale_flat[num_samples*(j-1)+u]
            ys[u] = z*scale+loc
        end

        # for u in 1:num_samples
        #     ys[u] = ys_std[u, t] * qx_scale[u, j] + qx_loc[u, j]
        # end

        x_mu_v = Vec{8,Float32}(0.0f0)
        y_mu_v = Vec{8,Float32}(0.0f0)
        for u in 1:8:num_samples_aligned
            x = vload(Vec{8,Float32}, xs_flat, coloff+u)
            y = vload(Vec{8,Float32}, ys, u)
            x_mu_v += x
            y_mu_v += y
        end
        x_mu = sum(x_mu_v)
        y_mu = sum(y_mu_v)

        for u in num_samples_aligned+1:num_samples
            x_mu += xs[u, t]
            y_mu += ys[u]
        end
        x_mu /= num_samples
        y_mu /= num_samples

        xy_cov_v = Vec{8, Float32}(0.0f0)
        x_var_v = Vec{8, Float32}(0.0f0)
        y_var_v = Vec{8, Float32}(0.0f0)
        for u in 1:8:num_samples_aligned
            x = vload(Vec{8,Float32}, xs_flat, coloff+u)
            y = vload(Vec{8,Float32}, ys, u)
            x_diff = x - x_mu
            y_diff = y - y_mu
            xy_cov_v += x_diff * y_diff
            x_var_v += x_diff * x_diff
            y_var_v += y_diff * y_diff
        end
        xy_cov = sum(xy_cov_v)
        x_var = sum(x_var_v)
        y_var = sum(y_var_v)

        for u in num_samples_aligned+1:num_samples
            x_diff = xs[u, t] - x_mu
            y_diff = ys[u] - y_mu
            xy_cov += x_diff * y_diff
            x_var += x_diff^2
            y_var += y_diff^2
        end

        ρ = xy_cov / (sqrt(x_var) * sqrt(y_var))
        miss_count += Int(abs(ρ) < c)
        if miss_count > max_miss_reps
            break
        end
    end

    return miss_count <= max_miss_reps
end


function write_graphviz(filename, edges, readable_labels)
    out = open(filename, "w")
    println(out, "graph coregulation {")
    println(out, "    node [shape=plaintext];")

    max_abs_ω = 0.0f0
    for (u, v, ω) in edges
        max_abs_ω = max(max_abs_ω, ω)
    end

    used_node_ids = Set{Int}()
    for (u, v, ω) in edges
        push!(used_node_ids, u)
        push!(used_node_ids, v)
    end

    for id in used_node_ids
        gene_name = readable_labels[id]
        println(out, "    node", id, " [label=\"", gene_name, "\"];")
    end

    for (u, v, ω) in edges
        println(
            out, "    node", u, " -- node", v,
            " [color=", ω > 0.0 ? "darkgoldenrod2" : "dodgerblue4",
            ", penwidth=", 1 + 4*abs(ω)/max_abs_ω,
            "];")
    end
    println(out, "}")
    close(out)
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
        Polee.gene_feature_matrix(
            ts, ts_metadata)

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

        # TODO: we need to reset the graph here to save space, but
        # have to re-add the likelihood approximation parameters.
        # Polee.tf[:reset_default_graph]()

        (num_features,
        splice_feature_idxs, splice_feature_transcript_idxs,
        # TODO: exclude MT, X, Y
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

    # splice_labels = [string("splice", i) for i in 1:size(qx_splice_loc, 2)]
    splice_labels = read_splice_feature_names(size(qx_splice_loc, 2))

    expression_mode = exp.(qx_loc)
    expression_mode ./= sum(expression_mode, dims=2)
    # gene_idx = maximum(expression_mode, dims=1)[1,:] .> 1e-5
    # gene_idx = maximum(expression_mode, dims=1)[1,:] .> 1e-5
    gene_idx = 1:size(expression_mode, 2)

    qx_loc_subset = qx_loc[:,gene_idx]
    qx_scale_subset = qx_scale[:,gene_idx]


    gene_names = gene_names[gene_idx]
    gene_ids = gene_ids[gene_idx]
    gene_names = [
        isempty(gene_name) ? gene_id : gene_name
        for (gene_id, gene_name) in zip(gene_ids, gene_names)]


    # splice_idx = minimum(qx_splice_scale, dims=1)[1,:] .< 0.2
    # splice_idx = minimum(qx_splice_scale, dims=1)[1,:] .< 0.2
    splice_idx = 1:size(qx_splice_scale, 2)

    qx_splice_loc_subset = qx_splice_loc[:,splice_idx]
    qx_splice_scale_subset = qx_splice_scale[:,splice_idx]

    splice_labels = splice_labels[splice_idx]

    @show size(qx_loc_subset)
    @show size(qx_splice_loc_subset)

    # Merge gene expression and splicing features



    # qx_merged_loc = hcat(qx_loc_subset, qx_splice_loc_subset)
    # qx_merged_scale = hcat(qx_scale_subset, qx_splice_scale_subset)

    # specific_labels = vcat(gene_ids, splice_labels)
    # readable_labels = vcat(gene_names, splice_labels)


    qx_merged_loc = qx_loc_subset
    qx_merged_scale = qx_scale_subset

    specific_labels = gene_ids
    readable_labels = gene_names

    # qx_merged_loc = Float32[
    #     -1  -1
    #     -1  -1
    #     -1  -1
    #     -1  -1
    #     -1  -1
    #     -1  -1
    #     -1  -1
    #     -1  -1
    #      1   1
    #      1   1
    #      1   1
    #      1   1
    #      1   1
    #      1   1
    #      1   1
    #      1   1 ]
    # qx_merged_scale = Float32[
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0
    #     1.0  1.0 ]

    # specific_labels = ["A", "B"]
    # readable_labels = ["A", "B"]


    # Shuffle so we can test on more interesting subsets
    Random.seed!(1234)
    idx = shuffle(1:size(qx_merged_loc, 2))
    qx_merged_loc = qx_merged_loc[:,idx]
    qx_merged_scale = qx_merged_scale[:,idx]
    specific_labels = specific_labels[idx]
    readable_labels = readable_labels[idx]

    # @time components = find_components(qx_merged_loc, qx_merged_scale)
    # sample_gaussian_graphical_model(qx_merged_loc, qx_merged_scale, components)


    # subset_idx = 1:30000
    # @time components = find_components(qx_merged_loc[:, subset_idx], qx_merged_scale[:, subset_idx])
    # edges = sample_gaussian_graphical_model(qx_merged_loc[:, subset_idx], qx_merged_scale[:, subset_idx], components)

    @time components = find_components(qx_merged_loc, qx_merged_scale)
    sample_gaussian_graphical_model(qx_merged_loc, qx_merged_scale, components)

    @show length(edges)
    write_graphviz("coexpression-graph.dot", edges, readable_labels)

    exit()
    #################### old stuff

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
        u += 1 # make 1-based
        for (v, lower, upper) in vs
            v += 1 # make 1-based
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
            println(corspearman(
                qx_merged_loc[:,u] .- mean(qx_merged_loc[:,u]),
                qx_merged_loc[:,v] .- mean(qx_merged_loc[:,v])))
        end
    end
    for id in used_node_ids
        gene_name = readable_labels[id]
        println(out, "    node", id, " [label=\"", gene_name, "\"];")
    end

    for (u, vs) in edges
        u += 1 # 1-based
        for (v, lower, upper) in vs
            v += 1 # 1-based
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
        u += 1 # make 1-based
        i = specific_labels[u]
        for (v, lower, upper) in vs
            v += 1 # make 1-based
            j = specific_labels[v]
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
