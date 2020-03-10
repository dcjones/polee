
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

@testset "Random tree heuristic" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--tree-method", "random",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

@testset "Sequential tree heuristic" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--tree-method", "sequential",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

@testset "Logistic Normal likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "logistic_normal",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

@testset "Kumaraswamy PTT likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "kumaraswamy_ptt",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

@testset "Logit Normal PTT likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "logit_normal_ptt",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

@testset "Normal ILR likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "normal_ilr",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

@testset "Normal ALR likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "normal_alr",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam" ]
    Polee.main(args)
end

# TODO: test polee sample

# TODO: test polee debug-sample

# TODO: some tests for Transcripts

# TODO: some tests for Reads

