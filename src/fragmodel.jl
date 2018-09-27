
"""
Model of conditional fragment probabilities. Should implement:

condfragprob(fm, t::Transcript, rs::Reads, alnpr::AlignmentPair, effective_length::Float32)
effective_length(fm, t::Transcript)
"""
abstract type FragModel; end


function fragment_length_prob(fm::FragModel, fraglen)
    return fraglen <= MAX_FRAG_LEN ? fm.fraglen_pmf[fraglen] : 0.0f0
end


function normal_pdf(mu, sd, x)
    invsqrt2pi = 0.3989422804014326779
    z = (x - mu) / sd
    return exp(-abs2(z)/2) * invsqrt2pi / sd
end


"""
A simple fragment model without any bias modeling.
"""
struct SimplisticFragModel <: FragModel
    fraglen_pmf::Vector{Float32}
    fraglen_cdf::Vector{Float32}
    fraglen_median::Int
    strand_specificity::Float32
end


function SimplisticFragModel(rs::Reads, ts::Transcripts)
    examples = rs.alignment_pairs

    # alignment pair fragment lengths
    alnpr_fraglen = Dict{UInt64, Int}()

    strand_match_count = 0
    strand_mismatch_count = 0

    for (t, alnpr) in eachoverlap(ts, examples)
        fraglen = fragmentlength(t, rs, alnpr)

        if fraglen === nothing
            continue
        end

        if alnpr.strand == t.strand
            strand_match_count += 1
        elseif alnpr.strand != STRAND_BOTH
            strand_mismatch_count += 1
        end

        if fraglen > 0
            h = hash(alnpr)
            if haskey(alnpr_fraglen, h)
                alnpr_fraglen[h] = min(alnpr_fraglen[h], fraglen)
            else
                alnpr_fraglen[h] = fraglen
            end
        end
    end

    strand_specificity = strand_match_count /
        (strand_match_count + strand_mismatch_count)
    negentropy = strand_specificity * log2(strand_specificity) +
        (1.0 - strand_specificity) * log2(1.0 - strand_specificity)
    println("Strand specificity: ", round(100.0 * (1 + negentropy), digits=1), "%")

    # compute fragment length frequencies
    fraglen_pmf = Vector{Float32}(undef, MAX_FRAG_LEN)
    fraglens_count = isempty(alnpr_fraglen) ?
        0 : sum(ifelse(fl <= MAX_FRAG_LEN, 1, 0) for fl in values(alnpr_fraglen))
    if fraglens_count < MIN_FRAG_LEN_COUNT
        # use fallback distribution
        for fl in 1:MAX_FRAG_LEN
            fraglen_pmf[fl] = normal_pdf(
                FALLBACK_FRAGLEN_MEAN, FALLBACK_FRAGLEN_SD, Float64(fl))
        end
        fraglen_pmf ./= sum(fraglen_pmf)
    else
        fill!(fraglen_pmf, 1.0f0)
        for fl in values(alnpr_fraglen)
            if fl <= MAX_FRAG_LEN
                fraglen_pmf[fl] += 1
            end
        end
        fraglen_pmf ./= sum(fraglen_pmf)
    end

    fraglen_cdf = copy(fraglen_pmf)
    for i in 2:length(fraglen_cdf)
        fraglen_cdf[i] += fraglen_cdf[i-1]
    end
    fraglen_median = searchsorted(fraglen_cdf, 0.5).start

    return SimplisticFragModel(
        fraglen_pmf, fraglen_cdf, fraglen_median, strand_specificity)
end

function compute_transcript_bias!(fm::SimplisticFragModel, ts::Transcripts)
end


function condfragprob(fm::SimplisticFragModel, t::Transcript, rs::Reads,
                      alnpr::AlignmentPair, effective_length::Float32)
    fraglen = fragmentlength(t, rs, alnpr)
    if fraglen === nothing
        return 0.0f0
    end

    # single-end read
    if fraglen <= 0
        aln = alnpr.metadata.mate1_idx > 0 ?
                    rs.alignments[alnpr.metadata.mate1_idx] : rs.alignments[alnpr.metadata.mate2_idx]
        if aln.flag == SAM.FLAG_REVERSE != 0
            max_frag_len = aln.rightpos - t.first + 1
        else
            max_frag_len = t.last - aln.leftpos + 1
        end

        fraglen = min(max_frag_len, fm.fraglen_median)
    end

    fragstrandpr = alnpr.strand == t.strand ?
        fm.strand_specificity : 1.0 - fm.strand_specificity

    fraglenpr = fragment_length_prob(fm, fraglen)

    fragpr = fragstrandpr * fraglenpr / effective_length

    return fragpr
end


function effective_length(fm::SimplisticFragModel, t::Transcript)
    tlen = exonic_length(t)
    el = 0.0f0
    for l in 1:min(tlen, MAX_FRAG_LEN)
        el += fm.fraglen_pmf[l] * (tlen - l + 1)
    end
    return Float32(max(el, MIN_EFFECTIVE_LENGTH))
end



mutable struct BiasedFragModel <: FragModel
    fraglen_pmf::Vector{Float32}
    fraglen_cdf::Vector{Float32}
    fraglen_median::Int
    high_prob_fraglens::Vector{Int}
    strand_specificity::Float32
    bias_model::BiasModel
end



function BiasedFragModel(
    rs::Reads, ts::Transcripts, read_assignments::Dict{Int, Int},
    dump_bias_training_examples::Bool;
    use_pos_bias::Bool=false)

    ts_by_id = collect(Transcript, ts)

    # strand-preference and fragment length distribution are estimated seperately
    # from other bias effects to simplify things. This way we can just randomly
    # shift reads without resizing fragments or changing strand.

    bias_foreground_examples = BiasTrainingExample[]
    bias_background_examples = BiasTrainingExample[]

    strand_match_count = 0
    strand_mismatch_count = 0
    fraglens = Int[]

    for alnpr in rs.alignment_pairs
        read_id = rs.alignments[alnpr.metadata.mate1_idx].id
        transcript_id = get(read_assignments, read_id, 0)
        if transcript_id == 0
            continue
        end
        t = ts_by_id[transcript_id]
        tseq = t.metadata.seq

        fragint = genomic_to_transcriptomic(t, rs, alnpr, FALLBACK_FRAGLEN_MEAN)
        fl = length(fragint)
        if fl <= 0
            continue
        end
        tpos = fragint.start

        if alnpr.strand == t.strand
            strand_match_count += 1
        elseif alnpr.strand != STRAND_BOTH
            strand_mismatch_count += 1
        end

        if fl <= 0 || tpos < 1 || tpos + fl - 1 > length(tseq)
            continue
        end

        # not enough sequence to train bias
        if fl < BIAS_SEQ_INNER_CTX + BIAS_SEQ_OUTER_CTX
            continue
        end

        # record the fragment length only if it was observed and not guessed
        if alnpr.metadata.mate1_idx != 0 && alnpr.metadata.mate2_idx != 0
            push!(fraglens, fl)
        end
        push!(bias_foreground_examples, BiasTrainingExample(tseq, tpos, fl))

        # perturb fragment position and record context as a background sample
        tpos = rand(1:length(tseq)-fl+1)
        # tpos = min(max(1, tpos + rand(-20:20)), length(tseq)-fl+1)
        push!(bias_background_examples, BiasTrainingExample(tseq, tpos, fl))
    end

    strand_specificity = strand_match_count /
        (strand_match_count + strand_mismatch_count)
    negentropy = strand_specificity * log2(strand_specificity) +
        (1.0 - strand_specificity) * log2(1.0 - strand_specificity)
    println("Strand specificity: ", round(100.0 * (1 + negentropy), digits=1), "%")

    # compute fragment length frequencies
    fraglen_pmf = Vector{Float32}(undef, MAX_FRAG_LEN)
    if length(fraglens) < MIN_FRAG_LEN_COUNT
        # use fallback distribution
        for fl in 1:MAX_FRAG_LEN
            fraglen_pmf[fl] = normal_pdf(
                FALLBACK_FRAGLEN_MEAN, FALLBACK_FRAGLEN_SD, fl)
        end
        fraglen_pmf ./= sum(fraglen_pmf)
    else
        fill!(fraglen_pmf, 1.0f0)
        for fl in fraglens
            if fl <= MAX_FRAG_LEN
                fraglen_pmf[fl] += 1
            end
        end
        fraglen_pmf ./= sum(fraglen_pmf)
    end

    open("fraglen_pmf.csv", "w") do output
        println(output, "fraglen,pr")
        for (fraglen, pr) in enumerate(fraglen_pmf)
            println(output, fraglen, ",", pr)
        end
    end

    fraglen_cdf = copy(fraglen_pmf)
    for i in 2:length(fraglen_cdf)
        fraglen_cdf[i] += fraglen_cdf[i-1]
    end
    fraglen_median = searchsorted(fraglen_cdf, 0.5).start

    p = shuffle(1:length(bias_foreground_examples))
    bias_foreground_examples = bias_foreground_examples[p]
    bias_background_examples = bias_background_examples[p]
    n_training = floor(Int, 0.8 * length(bias_foreground_examples))

    bias_model = BiasModel(
        ts, fraglen_pmf,
        bias_foreground_examples,
        bias_background_examples,
        use_pos_bias=use_pos_bias)

    # if dump_bias_training_examples
        open("bias-foreground.csv", "w") do output
            for example in bias_foreground_examples
                println(
                    output, DNASequence(example.left_seq),
                    ',', DNASequence(example.right_seq),
                    ',', example.frag_gc,
                    ',', example.tpdist,
                    ',', example.fpdist,
                    ',', example.tlen,
                    ',', example.flen,
                    ',', join(example.a_freqs, ','),
                    ',', join(example.c_freqs, ','),
                    ',', join(example.g_freqs, ','),
                    ',', join(example.t_freqs, ','))
            end
        end

        open("bias-background.csv", "w") do output
            for example in bias_background_examples
                println(
                    output, DNASequence(example.left_seq),
                    ',', DNASequence(example.right_seq),
                    ',', example.frag_gc,
                    ',', example.tpdist,
                    ',', example.fpdist,
                    ',', example.tlen,
                    ',', example.flen,
                    ',', join(example.a_freqs, ','),
                    ',', join(example.c_freqs, ','),
                    ',', join(example.g_freqs, ','),
                    ',', join(example.t_freqs, ','))
            end
        end
    # end
    # exit()

    fraglen_by_prob = collect(1:MAX_FRAG_LEN)[sortperm(fraglen_pmf, rev=true)]
    high_prob_fraglens = fraglen_by_prob[1:BIAS_EFFLEN_NUM_FRAGLENS]

    return BiasedFragModel(
        fraglen_pmf, fraglen_cdf, fraglen_median,
        high_prob_fraglens,
        strand_specificity,
        bias_model)
end


function compute_transcript_bias!(fm::BiasedFragModel, ts::Transcripts)
    ts_arr = collect(ts)
    Threads.@threads for t in ts_arr
    # for t in ts_arr
        compute_transcript_bias!(fm.bias_model, t)
    end

    # # codegen
    # compute_transcript_bias!(fm.bias_model, ts_arr[1])

    # @profile for t in ts_arr[1:20000]
    #     compute_transcript_bias!(fm.bias_model, t)
    # end
    # Profile.print()
    # exit()
end


function effective_length(fm::BiasedFragModel, t::Transcript)
    tseq = t.metadata.seq
    tlen = length(tseq)
    efflen = 0.0f0

    left_bias = t.metadata.left_bias
    right_bias = t.metadata.right_bias

    @inbounds for fraglen in fm.high_prob_fraglens
        if fraglen > tlen
            continue
        end

        fraglenpr = fragment_length_prob(fm, fraglen)

        gc_c = 1.0f0/fraglen
        frag_gc_prop = 0.0f0
        for pos in 1:fraglen
            nt = tseq[pos]
            frag_gc_prop += gc_c * isGC(nt)
        end

        c = 0f0
        for pos in 1:tlen-fraglen+1
            if pos > 1
                frag_gc_prop -= gc_c * isGC(tseq[pos-1])
                frag_gc_prop += gc_c * isGC(tseq[pos+fraglen-1])
            end

            c +=
                left_bias[pos] *
                right_bias[pos+fraglen-1] *
                evaluate(fm.bias_model.gc_model, frag_gc_prop)
        end
        efflen += c * fraglenpr
    end

    return Float32(max(efflen, MIN_EFFECTIVE_LENGTH))
end


function condfragprob(fm::BiasedFragModel, t::Transcript, rs::Reads,
                      alnpr::AlignmentPair, effective_length::Float32)
    tseq = t.metadata.seq
    tlen = length(tseq)
    fragint = genomic_to_transcriptomic(t, rs, alnpr, fm.fraglen_median)
    fraglen = length(fragint)
    if fraglen == 0 # incompatible fragment
        return 0.0f0
    end
    fraglenpr = fragment_length_prob(fm, fraglen)
    frag_gc_count = 0
    for pos in fragint
        frag_gc_count += isGC(tseq[pos])
    end
    frag_gc = frag_gc_count / length(fragint)

    # @show fragint
    # @show length(t.metadata.left_bias)
    # @show length(t.metadata.right_bias)

    fragbias =
        t.metadata.left_bias[fragint.start] *
        t.metadata.right_bias[fragint.stop] *
        evaluate(fm.bias_model.gc_model, frag_gc)

    fragstrandpr = alnpr.strand == t.strand ?
        fm.strand_specificity : 1.0 - fm.strand_specificity

    # @show (fragstrandpr, fraglenpr, fragbias, effective_length)
    fragpr = fragstrandpr * fraglenpr * fragbias / effective_length

    return fragpr
end
