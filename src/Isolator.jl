
module Isolator

using Bio.Align
using Bio.Intervals
using Bio.Seq
using Bio.StringFields
using ProgressMeter
using StatsBase

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
        update!(prog, position(reader.state.stream.source))

        if haskey(ts.transcripts.trees, entry.name)
            for t in ts.transcripts.trees[entry.name]
                seq = t.metadata.seq
                for exon in t.metadata.exons
                    append!(seq, entry.seq[exon.first:exon.last])
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

    rs = Reads("1.bam")
    ts = Transcripts("1.gff3")
    read_transcript_sequences!(ts, genome_filename)

    bm = BiasModel(rs, ts)
end


main()

end
