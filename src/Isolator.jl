
module Isolator

import TensorFlow
using Bio.Align
using Bio.Intervals
using Bio.Seq
using Bio.StringFields
using Distributions
using HDF5
using ProgressMeter
using StatsBase

include("constants.jl")
include("hattrie.jl")
include("transcripts.jl")
include("reads.jl")
include("bias.jl")
include("fragmodel.jl")
#include("outputformat_pb.jl")


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


function main()
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

    tic()
    for (t, alnpr) in intersect(ts, rs.alignment_pairs)
        intersection_candidate_count += 1
        fragpr = condfragprob(fm, t, rs, alnpr)
        if fragpr > 0.0
            j = alnpr.metadata.mate1_idx > 0 ?
                    rs.alignments[alnpr.metadata.mate1_idx].id :
                    rs.alignments[alnpr.metadata.mate2_idx].id
            push!(I, t.metadata.id)
            push!(J, j)
            push!(V, fragpr)
        end
    end
    toc()

    # TODO: combine function
    M = sparse(I, J, V)

    @show (M.m, M.n, length(M.nzval))

    #tic()
    #out = open("output.isolator_data", "w")
    #rec = Record(nzvalues=Float32[], js=UInt32[])
    #meta = ProtoBuf.meta(Record)
    #for j in 1:M.n
        #rec.nzvalues = unsafe_wrap(Vector{Float32},
                                   #pointer(M.nzval, M.colptr[j]),
                                   #M.colptr[j+1] - M.colptr[j])
        #rec.js = unsafe_wrap(Vector{UInt32},
                             #pointer(M.rowval, M.colptr[j]),
                             #M.colptr[j+1] - M.colptr[j])
        ## simplified version of writeproto(out, rec)
        #for attrib in meta.ordered
            #fld = attrib.fld
            #writeproto(out, getfield(rec, fld), attrib)
        #end
    #end
    #close(out)
    #toc()

    # TODO: order of reads outght to be shuffled (with fixed seed) just in case
    tic()
    h5open("output.h5", "w") do out
        out["m"] = M.m
        out["n"] = M.n
        out["colptr", "blosc", 3] = M.colptr
        out["rowval", "blosc", 3] = M.rowval
        out["nzval", "blosc", 3] = M.nzval
    end
    toc()

    # What we really need is to be able to read one "record" at a time. In this
    # case, one column. To avoid seeking around the file, shouldn't we be
    # storing records representing one column. (I guess reads should be rows,
    # technically, but it probably doesn't matter)

    # sampling procedure (sample a read) looks like
    #  1. read all of colptr
    #  2. choose a random column
    #  3. read data from nzval and rowval
end


main()

end
