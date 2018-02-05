
using Distributions, BioSequences, GenomicFeatures, ProgressMeter, HATTries, SHA

include("../../src/transcripts.jl")
include("../../src/sequences.jl")


function main()
    genome_filename, genes_filename = ARGS

    ts, ts_metadata = Transcripts(genes_filename)
    read_transcript_sequences!(ts, genome_filename)

    writer = FASTA.Writer(open("transcripts.fa", "w"), 80)
    for t in ts
        rec = FASTA.Record(t.metadata.name, t.metadata.seq)
        write(writer, rec)
    end
end

main()


