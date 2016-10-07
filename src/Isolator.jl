
module Isolator

using Bio.Align
using Bio.Intervals
using Bio.Seq
using Bio.StringFields
using Distributions
using ProgressMeter
using StatsBase
import TensorFlow

include("constants.jl")
include("hattrie.jl")
include("transcripts.jl")
include("reads.jl")
include("bias.jl")
include("fragmodel.jl")


function read_transcript_sequences!(ts, filename)
    prog = Progress(filesize(filename), 0.25, "Reading sequences ", 60)
    reader = open(FASTAReader, filename)
    entry = eltype(reader)()

    i = 0
    while !isnull(tryread!(reader, entry))
        if length(entry.seq) > 100000
            update!(prog, position(reader.state.stream.source))
        end

        if haskey(ts.transcripts.trees, entry.name)
            for t in ts.transcripts.trees[entry.name]
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
    reads_filename = "1.bam"
    transcripts_filename = "1.gff3"
    genome_filename = "/home/dcjones/data/homo_sapiens/seqs/1.fa"

    #reads_filename = "SRR948596.bam"
    #transcripts_filename = "/home/dcjones/data/homo_sapiens/Homo_sapiens.GRCh38.85.gff3"
    #genome_filename = "/home/dcjones/data/homo_sapiens/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

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
    for (t, alnpr) in intersect(ts.transcripts, rs.alignment_pairs)
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

    @show length(V)
    # TODO: serialize to HDF5
end


main()

end
