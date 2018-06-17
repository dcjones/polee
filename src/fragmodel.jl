
"""
Model of conditional fragment probabilities. Should implement:

condfragprob(fm, t::Transcript, rs::Reads, alnpr::AlignmentPair, effective_length::Float32)
effective_length(fm, t::Transcript)
"""
abstract type FragModel; end


function fragment_length_prob(fm::FragModel, fraglen)
    return fraglen <= MAX_FRAG_LEN ? fm.fraglen_pmf[fraglen] : 0.0f0
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
    alnpr_fraglen = ObjectIdDict()

    strand_match_count = 0
    strand_mismatch_count = 0

    for (t, alnpr) in eachoverlap(ts, examples)
        if alnpr.strand == t.strand
            strand_match_count += 1
        elseif alnpr.strand != STRAND_BOTH
            strand_mismatch_count += 1
        end

        # collect fragment lengths
        fraglen = fragmentlength(t, rs, alnpr)

        if !isnull(fraglen) && get(fraglen) > 0
            fl = get(fraglen)
            if haskey(alnpr_fraglen, alnpr)
                alnpr_fraglen[alnpr] = min(alnpr_fraglen[alnpr], fl)
            else
                alnpr_fraglen[alnpr] = fl
            end
        end
    end

    strand_specificity = strand_match_count /
        (strand_match_count + strand_mismatch_count)
    negentropy = strand_specificity * log2(strand_specificity) +
        (1.0 - strand_specificity) * log2(1.0 - strand_specificity)
    println("Strand specificity: ", round(100.0 * (1 + negentropy), 1), "%")

    # compute fragment length frequencies
    fraglen_pmf = Vector{Float32}(MAX_FRAG_LEN)
    fraglens_count = sum(ifelse(fl <= MAX_FRAG_LEN, 0, 1) for fl in values(alnpr_fraglen))
    if fraglens_count < MIN_FRAG_LEN_COUNT
        # use fallback distribution
        for fl in 1:MAX_FRAG_LEN
            fraglen_pmf[fl] = pdf(BIAS_FALLBACK_FRAGLEN_DIST, fl)
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
    fraglen_ = fragmentlength(t, rs, alnpr)
    if isnull(fraglen_)
        return 0.0f0
    end
    fraglen = get(fraglen_)

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

    fraglenpr = fragment_length_prob(fm, fraglen)
    fragpr = fraglenpr / effective_length

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



function BiasedFragModel(rs::Reads, ts::Transcripts, read_assignments::Dict{Int, Int})

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

        fragint = genomic_to_transcriptomic(t, rs, alnpr)
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

        push!(fraglens, fl)

        # fraqseq = extract_padded_seq(
        #     tseq, tpos - BIAS_SEQ_FRAG_PAD_LEFT, tpos + fl - 1 + BIAS_SEQ_FRAG_PAD_RIGHT)
        push!(bias_foreground_examples, BiasTrainingExample(tseq, tpos, fl))

        # perturb fragment position and record context as a background sample
        tpos = rand(1:length(tseq)-fl+1)
        # fraqseq = extract_padded_seq(
        #     tseq, tpos - BIAS_SEQ_FRAG_PAD_LEFT, tpos + fl - 1 + BIAS_SEQ_FRAG_PAD_RIGHT)
        push!(bias_background_examples, BiasTrainingExample(tseq, tpos, fl))
    end

    @show length(read_assignments)
    @show length(rs.alignment_pairs)
    @show length(fraglens)
    @show length(bias_foreground_examples)

    strand_specificity = strand_match_count /
        (strand_match_count + strand_mismatch_count)
    negentropy = strand_specificity * log2(strand_specificity) +
        (1.0 - strand_specificity) * log2(1.0 - strand_specificity)
    println("Strand specificity: ", round(100.0 * (1 + negentropy), 1), "%")

    # compute fragment length frequencies
    fraglen_pmf = Vector{Float32}(MAX_FRAG_LEN)
    if length(fraglens) < MIN_FRAG_LEN_COUNT
        # use fallback distribution
        for fl in 1:MAX_FRAG_LEN
            fraglen_pmf[fl] = pdf(BIAS_FALLBACK_FRAGLEN_DIST, fl)
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

    fraglen_cdf = copy(fraglen_pmf)
    for i in 2:length(fraglen_cdf)
        fraglen_cdf[i] += fraglen_cdf[i-1]
    end
    fraglen_median = searchsorted(fraglen_cdf, 0.5).start

    p = shuffle(1:length(bias_foreground_examples))
    bias_foreground_examples = bias_foreground_examples[p]
    bias_background_examples = bias_background_examples[p]
    n_training = floor(Int, 0.8 * length(bias_foreground_examples))

    bias_foreground_training_examples = bias_foreground_examples[1:n_training]
    bias_background_training_examples = bias_background_examples[1:n_training]
    bias_foreground_testing_examples = bias_foreground_examples[n_training+1:end]
    bias_background_testing_examples = bias_background_examples[n_training+1:end]

    bias_model = BiasModel(
        bias_foreground_training_examples,
        bias_background_training_examples,
        bias_foreground_testing_examples,
        bias_background_testing_examples)

    #=
    open("bias-foreground.csv", "w") do output
        for example in bias_foreground_examples
            println(
                output, example.left_seq,
                ',', example.right_seq,
                ',', example.frag_gc,
                ',', example.tpdist,
                ',', example.fpdist,
                ',', example.tlen,
                ',', example.fl,
                ',', join(example.a_freqs, ','),
                ',', join(example.c_freqs, ','),
                ',', join(example.g_freqs, ','),
                ',', join(example.t_freqs, ','))
        end
    end

    open("bias-background.csv", "w") do output
        for example in bias_background_examples
            println(
                output, example.left_seq,
                ',', example.right_seq,
                ',', example.frag_gc,
                ',', example.tpdist,
                ',', example.fpdist,
                ',', example.tlen,
                ',', example.fl,
                ',', join(example.a_freqs, ','),
                ',', join(example.c_freqs, ','),
                ',', join(example.g_freqs, ','),
                ',', join(example.t_freqs, ','))
        end
    end
    =#

    fraglen_by_prob = collect(1:MAX_FRAG_LEN)[sortperm(fraglen_pmf, rev=true)]
    high_prob_fraglens = fraglen_by_prob[1:BIAS_EFFLEN_NUM_FRAGLENS]
    @show high_prob_fraglens

    return BiasedFragModel(
        fraglen_pmf, fraglen_cdf, fraglen_median,
        high_prob_fraglens,
        strand_specificity,
        bias_model)
end


function compute_transcript_bias!(fm::BiasedFragModel, ts::Transcripts)
    ts_arr = collect(ts)
    Threads.@threads for t in ts_arr
        compute_transcript_bias!(fm.bias_model, t)
    end
end


function effective_length(fm::BiasedFragModel, t::Transcript)
    tseq = t.metadata.seq
    tlen = length(tseq)
    efflen = 0.0f0

    left_bias = t.metadata.left_bias
    right_bias = t.metadata.right_bias

    for fraglen in fm.high_prob_fraglens
        if fraglen > tlen
            continue
        end

        fraglenpr = fragment_length_prob(fm, fraglen)

        frag_gc_count = 0
        for pos in 1:fraglen
            nt = tseq[pos]
            frag_gc_count += isGC(nt)
        end

        c = 0f0
        for pos in 1:tlen-fraglen+1
            if pos > 1
                frag_gc_count -= isGC(tseq[pos-1])
                frag_gc_count += isGC(tseq[pos+fraglen-1])
            end

            c +=
                left_bias[pos] *
                right_bias[pos+fraglen-1] *
                evaluate(fm.bias_model.gc_model, Float32(frag_gc_count/fraglen))
        end
        efflen += c * fraglenpr
    end

    return efflen
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

    fragbias =
        t.metadata.left_bias[fragint.start] *
        t.metadata.right_bias[fragint.stop] *
        evaluate(fm.bias_model.gc_model, frag_gc)

    fragpr = fraglenpr * fragbias / effective_length

    return fragpr
end
