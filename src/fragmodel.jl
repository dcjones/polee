

type FragModel
    fraglen_pmf::Vector{Float64}
    fraglen_cdf::Vector{Float64}
    strand_specificity::Float64
    bm::BiasModel
end


"""
Randomly sample alignment pairs avoiding chosing duplicate reads
"""
function sample_training_examples(rs::Reads, n::Int)
    n = min(n, length(rs.alignment_pairs))
    examples = IntervalCollection{AlignmentPairMetadata}()

    # count unique alignment start positions, ignoring sequence name for
    # simplicity
    starts = IntSet()
    for alnpr in rs.alignment_pairs
        push!(starts, alnpr.first)
    end
    @show length(starts)
    @show maximum(starts)

    starts_subset = IntSet()
    starts_subset_idxs = IntSet(sample(1:length(starts), n, replace=false))
    for (i, start) in enumerate(starts)
        if i in starts_subset_idxs
            push!(starts_subset, start)
        end
    end

    last_start = 0
    for alnpr in rs.alignment_pairs
        if alnpr.first != last_start
            if alnpr.first in starts_subset
                push!(examples, alnpr)
            end
            last_start = alnpr.first
        end
    end

    return examples
end


function FragModel(rs::Reads, ts::Transcripts, n::Int=100000,
                   bias_upctx::Int=15, bias_downctx::Int=15)
    examples = sample_training_examples(rs, n)

    # TODO: positional bias
    # TODO: strand bias

    # alignment pair fragment lengths
    alnpr_fraglen = ObjectIdDict()

    # sequence bias training examples
    fs_foreground = DNASequence[]
    fs_background = DNASequence[]
    ss_foreground = DNASequence[]
    ss_background = DNASequence[]

    strand_match_count = 0
    strand_mismatch_count = 0

    for (t, alnpr) in intersect(ts.transcripts, examples)
        # collect sequences for sequence bias
        if alnpr.metadata.mate1_idx > 0
            push_alignment_context!(
                fs_foreground, fs_background, ss_foreground, ss_background,
                bias_upctx, bias_downctx, rs,
                rs.alignments[alnpr.metadata.mate1_idx], t)
        end

        if alnpr.metadata.mate2_idx > 0
            push_alignment_context!(
                fs_foreground, fs_background, ss_foreground, ss_background,
                bias_upctx, bias_downctx, rs,
                rs.alignments[alnpr.metadata.mate2_idx], t)
        end

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

    # train sequence bias models
    println("training first-strand bias model")
    fs_model = train_model(fs_foreground, fs_background,
                           bias_upctx + 1 + bias_downctx)
    println("training second-strand bias model")
    ss_model = train_model(ss_foreground, ss_background,
                           bias_upctx + 1 + bias_downctx)

    bm = BiasModel(bias_upctx, bias_downctx,
                   fs_foreground, fs_background,
                   ss_foreground, ss_background,
                   fs_model, ss_model)
    write_statistics(open("bias.csv", "w"), bm)

    strand_specificity = strand_match_count /
        (strand_match_count + strand_mismatch_count)
    negentropy = strand_specificity * log2(strand_specificity) +
        (1.0 - strand_specificity) * log2(1.0 - strand_specificity)
    println("Strand specificity: ", round(100.0 * (1 + negentropy), 1), "%")

    # compute fragment length frequencies
    fraglen_pmf = fill(1.0, MAX_FRAG_LEN) # init with pseudocount
    for fl in values(alnpr_fraglen)
        if fl <= MAX_FRAG_LEN
            fraglen_pmf[fl] += 1
        end
    end
    fraglen_pmf ./= sum(fraglen_pmf)
    out = open("fraglen.csv", "w")
    println(out, "fraglen,freq")
    for (fl, freq) in enumerate(fraglen_pmf)
        println(out, fl, ",", freq)
    end

    fraglen_cdf = copy(fraglen_pmf)
    for i in 2:length(fraglen_cdf)
        fraglen_cdf[i] += fraglen_cdf[i-1]
    end

    return FragModel(fraglen_pmf, fraglen_cdf, strand_specificity, bm)
end


function condfragprob(fm::FragModel, t::Transcript, rs::Reads,
                      alnpr::AlignmentPair)
    fraglen_ = fragmentlength(t, rs, alnpr)
    if isnull(fraglen_)
        return 0.0
    end
    fraglen = get(fraglen_)

    # single-end read
    if fraglen <= 0
        # TODO: decide what to do about this
    end

    # TODO: look at FragWeightEstimationThread::fragment_weight for how to
    # handle all this shit

    fragpr = 1.0
    fragpr *= fraglen <= length(fm.fraglen_pmf) ? fm.fraglen_pmf[fraglen] : 0.0

    tlen = exonic_length(t)
    fragpr /= tlen <= length(fm.fraglen_cdf) ? fm.fraglen_cdf[tlen] : 1.0

    return fragpr
end


