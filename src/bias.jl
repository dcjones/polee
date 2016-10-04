

type BiasModel
end


function perturb_transcriptomic_position(pos, lower, upper)
    @assert lower <= pos <= upper
    pos_ = pos + round(Int, rand(Normal(5,5)))
    pos_ = max(lower, min(upper, pos_))
    return pos_
end


function push_alignment_context!(
            fs_foreground, fs_background, ss_foreground, ss_background,
            upctx, downctx, reads::Reads, aln::Alignment, t::Transcript)

    tseq = t.metadata.seq

    strand = ifelse(aln.flag & SAM_FLAG_REVERSE == 0, STRAND_POS, STRAND_NEG)
    pos = genomic_to_transcriptomic(t,
                ifelse(strand == STRAND_POS, aln.leftpos, aln.rightpos))
    if pos < 1
        return false, false
    end

    # adjust position in the presense of soft-clipping
    if cigar_len(aln) > 1
        leading_clip, trailing_clip = 0, 0
        for (i, (op, len)) in enumerate(CigarIter(reads, aln))
            if i == 1 && op == OP_SOFT_CLIP
                leading_clip = len
            else
                if op == OP_SOFT_CLIP
                    trailing_clip = len
                else
                    trailing_clip = 0
                end
            end
        end

        if strand == STRAND_POS
            pos -= leading_clip
        else
            pos += trailing_clip
        end
    end

    leftctx, rightctx =
        ifelse(strand == STRAND_POS, (upctx, downctx), (downctx, upctx))

    if pos - leftctx < 1 || pos + rightctx > length(tseq)
        return false, false
    end
    pos_ = perturb_transcriptomic_position(
            pos, 1 + leftctx, length(tseq) - rightctx)

    ctxseq = tseq[pos-leftctx:pos+rightctx]
    ctxseq_ = tseq[pos_-leftctx:pos_+rightctx]

    if t.strand == STRAND_NEG
        reverse_complement!(ctxseq)
        reverse_complement!(ctxseq_)
    end
    # TODO: fix up sequences with non-ACTGs
    # really we could just do this when we do the 1-hot encoding

    if strand == t.strand
        push!(ss_foreground, ctxseq)
        push!(ss_background, ctxseq_)
    else
        push!(fs_foreground, ctxseq)
        push!(fs_background, ctxseq_)
    end

    return true, strand == t.strand
end


function BiasModel(reads::Reads, transcripts::Transcripts,
                   upctx::Integer=5, downctx::Integer=5,
                   n::Integer=100000, matched_examples::Bool=true)
    # sample read pair ends
    n = min(n, length(reads.alignment_pairs))
    # TODO: how can I sample avoiding reads for which there are many many
    # copies? Is this a huge issue in practice?
    # Idea: sample with replacement. Reject alignments at seen positions.
    # that way we can control how many examples we get
    exidxs = Set(sample(1:length(reads.alignment_pairs), n, replace=false))
    examples = IntervalCollection{AlignmentPairMetadata}()
    for (i, alnpr) in enumerate(reads.alignment_pairs)
        if i in exidxs
            push!(examples, alnpr)
        end
    end

    # we collect separate sequence contexts for ends corresponding to first- and
    # second-strand cDNA synthesis.
    seen_aln_pairs = ObjectIdDict()
    fs_foreground = DNASequence[]
    fs_background = DNASequence[]
    ss_foreground = DNASequence[]
    ss_background = DNASequence[]

    # find compatible representative transcripts
    for (t, ap) in intersect(transcripts.transcripts, examples)
        if get(seen_aln_pairs, ap, false)
            continue
        end

        mate1_used, mate1_second_strand = push_alignment_context!(
            fs_foreground, fs_background, ss_foreground, ss_background,
            upctx, downctx, reads, reads.alignments[ap.metadata.mate1_idx], t)

        # single-end
        if ap.metadata.mate1_idx == ap.metadata.mate2_idx
            if matched_examples && mate1_used
                if mate1_second_strand
                    pop!(ss_foreground)
                    pop!(ss_background)
                else
                    pop!(fs_foreground)
                    pop!(fs_background)
                end
            end

            continue
        end

        if matched_examples && !mate1_used
            continue
        end

        mate2_used, mate2_second_strand = push_alignment_context!(
            fs_foreground, fs_background, ss_foreground, ss_background,
            upctx, downctx, reads, reads.alignments[ap.metadata.mate2_idx], t)

        if matched_examples && !mate2_used
            if mate1_second_strand
                pop!(ss_foreground)
                pop!(ss_background)
            else
                pop!(fs_foreground)
                pop!(fs_background)
            end
        end

        seen_aln_pairs[ap] = mate1_used | mate2_used
    end

    @show length(ss_foreground)
    @show length(fs_foreground)
    @show length(ss_background)
    @show length(fs_background)

    #for seq in fs_foreground
        #@show seq
    #end

    ss_freqs = Dict{DNANucleotide, Float64}()
    for seq in ss_foreground
        c = seq[upctx+1]
        if haskey(ss_freqs, c)
            ss_freqs[c] += 1
        else
            ss_freqs[c] = 1
        end
    end

    for k in keys(ss_freqs)
        ss_freqs[k] /= length(ss_foreground)
    end

    fs_freqs = Dict{DNANucleotide, Float64}()
    for seq in fs_foreground
        c = seq[upctx+1]
        if haskey(fs_freqs, c)
            fs_freqs[c] += 1
        else
            fs_freqs[c] = 1
        end
    end

    for k in keys(fs_freqs)
        fs_freqs[k] /= length(fs_foreground)
    end

    dinuc_freqs = Dict{Tuple{DNANucleotide, DNANucleotide}, Float64}()
    for (a, b) in zip(ss_foreground, fs_foreground)
        k = (a[upctx+1], b[upctx+1])
        if haskey(dinuc_freqs, k)
            dinuc_freqs[k] += 1
        else
            dinuc_freqs[k] = 1
        end
    end

    for k in keys(dinuc_freqs)
        dinuc_freqs[k] /= length(ss_foreground)
    end

    for ((u,v),f) in dinuc_freqs
        if haskey(ss_freqs, u) && haskey(fs_freqs, v)
            @show (u, v, round(f, 3), round(ss_freqs[u] * fs_freqs[v], 3))
        end
    end
end

