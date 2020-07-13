
using Test
using Polee


@testset "Likelihood approximation w/ genome alignments" begin
    args = String[
        "prep-sample",
        "dataset/genome.fa",
        "dataset/mBr_M_6w_1.genome.bam",
        "dataset/annotations.gff3",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Likelihood approximation w/ transcriptome alignments" begin
    args = String[
        "prep-sample",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Random tree heuristic" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--tree-method", "random",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Sequential tree heuristic" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--tree-method", "sequential",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Logistic Normal likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "logistic_normal",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Kumaraswamy PTT likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "kumaraswamy_ptt",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Logit Normal PTT likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "logit_normal_ptt",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Normal ILR likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "normal_ilr",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Normal ALR likelihood approximation" begin
    args = String[
        "prep-sample",
        "--no-bias",
        "--approx-method", "normal_alr",
        "dataset/transcriptome.fa",
        "dataset/mBr_M_6w_1.transcriptome.bam",
        "-o", "output/prep.h5" ]
    Polee.main(args)
end

@testset "Sampling from approximate likelihood" begin
    args = String[
        "sample",
        "--num-samples", "5",
        "--annotations", "dataset/annotations.gff3",
        "dataset/mBr_M_6w_1.prep.h5",
        "-o", "output/sample.csv"]
    Polee.main(args)
end

@testset "Outputing kallisto format" begin
    args = String[
        "sample",
        "--num-samples", "5",
        "--annotations", "dataset/annotations.gff3",
        "dataset/mBr_M_6w_1.prep.h5",
        "--kallisto",
        "-o", "output/sample.h5"]
    Polee.main(args)
end

@testset "Gibbs sampler" begin
    args = String[
        "debug-sample",
        "--burnin", "20",
        "--num-samples", "20",
        "--stride", "2",
        "--annotations", "dataset/annotations.gff3",
        "dataset/mBr_M_6w_1.likelihood-matrix.h5",
        "-o", "output/gibbs.csv"]
    Polee.main(args)
end

@testset "EM" begin
    args = String[
        "debug-optimize",
        "--annotations", "dataset/annotations.gff3",
        "dataset/mBr_M_6w_1.likelihood-matrix.h5",
        "-o", "output/em.csv"]
    Polee.main(args)
end

@testset "Transcript Regression" begin
    args = String[
        "model", "regression",
        "--feature", "transcripts",
        "--factors", "tissue",
        "dataset/experiment.yml"]
    Polee.main(args)
end

# TODO: some tests for Transcripts

# TODO: some tests for Reads

