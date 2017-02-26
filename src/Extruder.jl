#!/usr/bin/env julia

module Extruder

using ArgParse
using Bio.Align
using Bio.Intervals
using Bio.Seq
using Bio.StringFields
using Distributions
using HDF5
using ProgressMeter
using PyCall
using StatsBase
import YAML
import TensorFlow

include("fastmath.jl")
using .FastMath

include("rsb.jl")
using .RSB
RSB.rsb_init()

include("constants.jl")
include("hattrie.jl")
include("transcripts.jl")
include("reads.jl")
include("bias.jl")
include("fragmodel.jl")
include("model.jl")
include("sample.jl")
include("likelihood-approximation.jl")
include("estimate.jl")


function read_transcript_sequences!(ts, filename)
    prog = Progress(filesize(filename), 0.25, "Reading sequences ", 60)
    reader = open(FASTAReader, filename)
    entry = eltype(reader)()

    i = 0
    while !isnull(tryread!(reader, entry))
        if length(entry.seq) > 100000
            update!(prog, position(reader.state.stream.source))
        end

        if haskey(ts.trees, entry.name)
            for t in ts.trees[entry.name]
                seq = t.metadata.seq
                for exon in t.metadata.exons
                    if exon.last <= length(entry.seq)
                        append!(seq, entry.seq[exon.first:exon.last])
                    end
                end
            end
        end
    end
    finish!(prog)
end


function print_usage()
    println("Usage: extruder <command>\n")
    println("where command is one of:")
    println("  likelihood-matrix")
    println("  likelihood-approx")
    println("  prepare-sample")
    println("  estimate")
end


function main()
    if isempty(ARGS)
        print_usage()
        exit(1)
    end

    subcmd = ARGS[1]
    subcmd_args = ARGS[2:end]
    arg_settings = ArgParseSettings()

    if subcmd == "likelihood-matrix"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "likelihood-matrix.h5"
            "transcripts_filename"
                required = true
            "genome_filename"
                required = true
            "reads_filename"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              output=Nullable(parsed_args["output"]))
        return

    elseif subcmd == "likelihood-approx"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "likelihood_matrix_filename"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        approximate_likelihood(parsed_args["likelihood_matrix_filename"],
                               parsed_args["output"])
        return

    elseif subcmd == "prepare-sample"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "transcripts_filename"
                required = true
            "genome_filename"
                required = true
            "reads_filename"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"])
        approximate_likelihood(sample, parsed_args["output"])
        return

    elseif subcmd == "estimate"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "results.h5"
            "prepared_sample"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        estimate(parsed_args["prepared_sample"],
                 parsed_args["output"])
        return
    elseif subcmd == "likelihood-approx-from-isolator"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "isolator_matrix"
                required = true
            "isolator_effective_lengths"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        approximate_likelihood_from_isolator(
                               parsed_args["isolator_matrix"],
                               parsed_args["isolator_effective_lengths"],
                               parsed_args["output"])
        return
    else
        println("Unknown command: ", subcmd, "\n")
        print_usage()
        exit(1)
    end
end


main()

end
