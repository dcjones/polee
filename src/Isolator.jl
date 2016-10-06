
module Isolator

using Bio.Align
using Bio.Intervals
using Bio.Seq
using Bio.StringFields
using Distributions
using ProgressMeter
using StatsBase
import TensorFlow

include("hattrie.jl")
include("transcripts.jl")
include("reads.jl")
include("bias.jl")


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
    #bm = BiasModel(rs, ts)
    #write_statistics(open("bias.csv", "w"), bm)

    println("intersecting...")

    # sparse matrix indexes and values
    I = UInt32[]
    J = UInt32[]
    V = Float32[]
    intersection_count = 0
    intersection_candidate_count = 0

    # Multi-threaded version seems to suck
    #buflen = 100000
    #intersect_buffer = Array(Tuple{Transcript, AlignmentPair}, buflen)
    #bufcnt = 0

    #function process_buffer(rs, intersect_buffer)
        #Threads.@threads for item in intersect_buffer
            #t, alnpr = item
            #fragmentlength(t, rs, alnpr)
        #end

    #end

    #tic()
    #for (t, alnpr) in intersect(ts.transcripts, rs.alignment_pairs)
        #intersection_count += 1
        #intersect_buffer[bufcnt += 1] = (t, alnpr)
        #if bufcnt == buflen
            #process_buffer(rs, intersect_buffer)
            #bufcnt = 0
        #end
    #end

    #if bufcnt > 0
        #process_buffer(rs, intersect_buffer)
        #bufcnt = 0
    #end
    #toc()

    tic()
    for (t, alnpr) in intersect(ts.transcripts, rs.alignment_pairs)
        intersection_candidate_count += 1
        fraglen = fragmentlength(t, rs, alnpr)
        if !isnull(fraglen)
            intersection_count += 1
        end
    end
    toc()

    @show intersection_count
    @show intersection_candidate_count
end


main()

end
