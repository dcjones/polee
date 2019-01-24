
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
using Statistics

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

    # Fit heirarchical model (of transcript expression)
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

    # TRY: lets just consider the stuff with high expression for now
    idx = maximum(qx_loc, dims=1)[1,:] .> -10.0

    @show quantile(reshape(qx_scale, (length(qx_scale),)), Float32[0.1, 0.5, 0.9])

    @show sum(idx)
    # qx_loc = qx_loc[:,idx]
    # qx_scale = qx_scale[:,idx]

    @show quantile(reshape(qx_scale, (length(qx_scale),)), Float32[0.1, 0.5, 0.9])

    # Fit covariance matrix column by column
    coregulation_py.estimate_gmm_precision(
        qx_loc, qx_scale)
end

main()