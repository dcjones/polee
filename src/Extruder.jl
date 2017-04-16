#!/usr/bin/env julia

module Extruder

using ArgParse
using Bio.Align
using Bio.Intervals
using Bio.Seq
using Bio.StringFields
using Distributions
using HATTries
using HDF5
using ProgressMeter
using PyCall
using RecursiveSparseBlocks
using SQLite
using StatsBase
import SHA
import YAML
import TensorFlow

include("fastmath.jl")
using .FastMath

include("constants.jl")
include("transcripts.jl")
include("reads.jl")
include("bias.jl")
include("fragmodel.jl")
include("model.jl")
include("sample.jl")
include("likelihood-approximation.jl")
include("estimate.jl")
include("hattrie_stringfield.jl")

# TODO: automate including everything under models
EXTRUDER_MODELS = Dict{String, Function}()
include("models/linear-regression.jl")
include("models/simple-linear-regression.jl")
include("models/simple-mode.jl")


function read_transcript_sequences!(ts, filename)
    if endswith(filename, ".2bit")
        read_transcript_sequences_from_twobit!(ts, filename)
    else
        read_transcript_sequences_from_fasta!(ts, filename)
    end
end


function read_transcript_sequences_from_fasta!(ts, filename)
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


function read_transcript_sequences_from_twobit!(ts, filename)
    reader = open(TwoBitReader, filename)
    prog = Progress(length(ts.trees), 0.25, "Reading sequences ", 60)

    for (i, (name, tree)) in enumerate(ts.trees)
        update!(prog, i)
        local refseq
        try
            refseq = reader[name].seq
        catch
            continue
        end

        for t in tree
            seq = t.metadata.seq
            for exon in t.metadata.exons
                if exon.last <= length(refseq)
                    append!(seq, DNASequence(refseq[exon.first:exon.last]))
                end
            end
        end
    end
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
            "--excluded-seqs"
                required = false
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        excluded_seqs = Set{String}()
        if parsed_args["excluded-seqs"] != nothing
            open(parsed_args["excluded-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chop(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs,
                              Nullable(parsed_args["output"]))
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
            "--excluded-seqs"
                required = false
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        excluded_seqs = Set{String}()
        if parsed_args["excluded-seqs"] != nothing
            open(parsed_args["excluded-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chop(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs)
        approximate_likelihood(sample, parsed_args["output"])
        return

    elseif subcmd == "estimate"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "output.txt"
            "model"
                required = true
            "transcripts"
                required = true
            "experiment"
                required = true
        end

        parsed_args = parse_args(subcmd_args, arg_settings)

        # automate generating serialized transcripts when they don't exist
        ts, metadata = Transcripts(parsed_args["transcripts"])
        write_transcripts("genes.db", ts, metadata)

        EXTRUDER_MODELS[parsed_args["model"]](
                 parsed_args["experiment"],
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
