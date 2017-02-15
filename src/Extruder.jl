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
using StatsBase
using Stan
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
    println("  prepare")
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
            "transcripts_file"
                required = true
            "genome_filename"
                required = true
            "reads_filename"
                required = true
        end
        parsed_args = parsed_args(subcmd_args, arg_settings)
        error("TODO")

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
    elseif subcmd == "prepare"
        # TODO: do both likelihood-matrix and likelihood-approx in one step with
        # no intermediate output

        error("TODO")
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
            "isolator_data"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        approximate_likelihood_from_isolator(
                               parsed_args["isolator_data"],
                               parsed_args["output"])
        return
    else
        println("Unknown command: ", subcmd, "\n")
        print_usage()
        exit(1)
    end


    # TODO: put all this stuff under likelihood-matrix

    #reads_filename = "MT.bam"
    #transcripts_filename = "MT.gff3"
    #genome_filename = "/home/dcjones/data/homo_sapiens/seqs/MT.fa"

    #reads_filename = "1.bam"
    #transcripts_filename = "1.gff3"
    #genome_filename = "/home/dcjones/data/homo_sapiens/seqs/1.fa"

    reads_filename = "SRR948596.bam"
    transcripts_filename = "/home/dcjones/data/homo_sapiens/Homo_sapiens.GRCh38.85.gff3"
    genome_filename = "/home/dcjones/data/homo_sapiens/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

    rs = Reads(reads_filename)
    ts = Transcripts(transcripts_filename)
    read_transcript_sequences!(ts, genome_filename)
    fm = FragModel(rs, ts)

    println("intersecting...")

    # sparse matrix indexes and values
    I = UInt32[]
    J = UInt32[]
    V = Float32[]
    intersection_count = 0
    intersection_candidate_count = 0

    MIN_FRAG_PROB = 1e-8

    # reassign indexes to alignments to group by position
    aln_idx_map = Dict{Int, Int}()
    for alnpr in rs.alignment_pairs
        if alnpr.metadata.mate1_idx > 0
            get!(aln_idx_map, rs.alignments[alnpr.metadata.mate1_idx].id,
                 length(aln_idx_map) + 1)
        else
            get!(aln_idx_map, rs.alignments[alnpr.metadata.mate2_idx].id,
                 length(aln_idx_map) + 1)
        end
    end

    # reassign transcript indexes to group by position
    for (tid, t) in enumerate(ts)
        t.metadata.id = tid
    end

    tic()
    for (t, alnpr) in intersect(ts, rs.alignment_pairs)
        intersection_candidate_count += 1
        fragpr = condfragprob(fm, t, rs, alnpr)
        if fragpr > MIN_FRAG_PROB
            i = alnpr.metadata.mate1_idx > 0 ?
                    rs.alignments[alnpr.metadata.mate1_idx].id :
                    rs.alignments[alnpr.metadata.mate2_idx].id
            push!(I, aln_idx_map[i])
            push!(J, t.metadata.id + 1) # +1 to make room for pseudotranscript
            push!(V, fragpr)
        end
    end

    # conditional probability of observing a fragment given it belongs to some
    # other unknown transcript or something else. TODO: come up with some
    # principled number of this.
    const SPURIOSITY_PROB = MIN_FRAG_PROB
    for i in 1:maximum(I)
        push!(I, i)
        push!(J, 1)
        push!(V, SPURIOSITY_PROB)
    end
    toc()

    # TODO: combine function (or is + reasonable?)
    M = sparse(I, J, V)

    @show (M.m, M.n, length(M.nzval))

    # TODO: order of reads outght to be shuffled (with fixed seed) just in case
    tic()
    h5open("output.h5", "w") do out
        out["m"] = M.m
        out["n"] = M.n
        out["colptr", "compress", 1] = M.colptr
        out["rowval", "compress", 1] = M.rowval
        out["nzval", "compress", 1] = M.nzval
    end
    toc()
end


main()

end
