

type BiasModel
    upctx::Int
    downctx::Int
    fs_foreground::Vector{DNASequence}
    fs_background::Vector{DNASequence}
    ss_foreground::Vector{DNASequence}
    ss_background::Vector{DNASequence}
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

    if strand == t.strand
        push!(ss_foreground, ctxseq)
        push!(ss_background, ctxseq_)
    else
        reverse_complement!(ctxseq)
        reverse_complement!(ctxseq_)
        push!(fs_foreground, ctxseq)
        push!(fs_background, ctxseq_)
    end

    return true, strand == t.strand
end


function BiasModel(reads::Reads, transcripts::Transcripts,
                   upctx::Integer=15, downctx::Integer=25,
                   n::Integer=100000)
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
            continue
        end

        mate2_used, mate2_second_strand = push_alignment_context!(
            fs_foreground, fs_background, ss_foreground, ss_background,
            upctx, downctx, reads, reads.alignments[ap.metadata.mate2_idx], t)

        seen_aln_pairs[ap] = mate1_used | mate2_used
    end

    return BiasModel(upctx, downctx,
                     fs_foreground, fs_background,
                     ss_foreground, ss_background)
end


function write_statistics(out::IO, bm::BiasModel)
    println(out, "strand,state,position,nucleotide,frequency")
    example_collections = [
        ("first-strand", "foreground", bm.fs_foreground),
        ("first-strand", "background", bm.fs_background),
        ("second-strand", "foreground", bm.ss_foreground),
        ("second-strand", "background", bm.ss_background)]
    freqs = Dict{DNANucleotide, Float64}(
        DNA_A => 0.0, DNA_C => 0.0, DNA_G => 0.0, DNA_T => 0.0)

    n = bm.upctx + 1 + bm.downctx
    for (strand, state, examples) in example_collections
        for pos in 1:n
            for k in keys(freqs)
                freqs[k] = 0.0
            end

            for seq in examples
                if seq[pos] != DNA_N
                    freqs[seq[pos]] += 1
                end
            end

            z = sum(values(freqs))
            for (k, v) in freqs
                @printf(out, "%s,%s,%d,%c,%0.4f\n",
                        strand, state, pos - bm.upctx - 1, Char(k), v/z)
            end
        end
    end
end

