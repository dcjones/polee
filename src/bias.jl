

type BiasModel
end


function perturb_transcriptomic_position(pos, lower, upper)
    @assert lower <= pos <= upper
    # TODO
    return pos
end


function push_alignment_context!(
            fs_foreground, fs_background, ss_foreground, ss_background,
            upctx, downctx, reads::Reads, aln::Alignment, t::Transcript)

    tseq = t.metadata.seq

    strand = ifelse(aln.flag & SAM_FLAG_REVERSE == 0, STRAND_POS, STRAND_NEG)
    pos = genomic_to_transcriptomic(t,
                ifelse(strand == STRAND_POS, aln.leftpos, aln.rightpos))
    if pos < 1
        return false
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
        return false
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

    return true
end


function BiasModel(reads::Reads, transcripts::Transcripts,
                   upctx::Integer=5, downctx::Integer=15,
                   n::Integer=10000)
    # sample read pair ends
    n = min(n, length(reads.alignment_pairs))
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

        used = push_alignment_context!(
            fs_foreground, fs_background, ss_foreground, ss_background,
            upctx, downctx, reads, reads.alignments[ap.metadata.mate1_idx], t)

        # single-end
        if ap.metadata.mate1_idx == ap.metadata.mate2_idx
            continue
        end

        used |= push_alignment_context!(
            fs_foreground, fs_background, ss_foreground, ss_background,
            upctx, downctx, reads, reads.alignments[ap.metadata.mate2_idx], t)

        seen_aln_pairs[ap] = used
    end

    @show length(ss_foreground)
    @show length(fs_foreground)
    @show length(ss_background)
    @show length(fs_background)

    #@show ss_foreground
    for seq in ss_foreground
        @show seq
    end
    # TODO: what the fuck. I'm seeing a bunch of gap characters.
end

