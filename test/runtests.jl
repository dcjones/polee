
using Test
using Polee


@testset "Likelihood approximation w/ genome alignments" begin
    args = String[
        "prep-sample",
        "dataset/genome.fa",
        "dataset/mBr_M_6w_1.genome.bam",
        "dataset/annotations.gff3" ]
    Polee.main(args)
end


@testset "Likelihood approximation w/ transcriptome alignments" begin
    args = String[
        "prep-sample",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

@testset "Sampling from approximated likelihood" begin
end

# TODO: test polee sample

# TODO: test polee debug-sample

# TODO: some tests for Transcripts

# TODO: some tests for Reads

