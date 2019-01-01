
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

    max_num_samples = nothing
    batch_size = nothing
    loaded_samples = Polee.load_samples_from_specification(
        spec, ts, ts_metadata, max_num_samples, batch_size)


    # Fit heirarchical model (of transcript expression)
    num_samples, n = size(loaded_samples.x0_values)
    x0_log = log.(loaded_samples.x0_values)
    qx_loc, qx_scale =
        Polee.polee_py[:estimate_transcript_expression](
            loaded_samples.init_feed_dict, num_samples, n,
            loaded_samples.variables, x0_log)
    Polee.tf[:reset_default_graph]()

    # Fit covariance matrix column by column
    coregulation_py.estimate_gmm_precision(
        qx_loc, qx_scale)
end

main()