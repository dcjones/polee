


struct BiasTrainingExample
    left_seq::DNASequence
    right_seq::DNASequence
    frag_gc::Float64
    tpdist::Float64
    tlen::Float64
    fl::Float64
    a_freqs::NTuple{BIAS_NUM_FREQ_BINS, Float64}
    c_freqs::NTuple{BIAS_NUM_FREQ_BINS, Float64}
    g_freqs::NTuple{BIAS_NUM_FREQ_BINS, Float64}
    t_freqs::NTuple{BIAS_NUM_FREQ_BINS, Float64}
end


function BiasTrainingExample(tseq, tpos, fl)
    fragseq = extract_padded_seq(
        tseq, tpos - BIAS_SEQ_OUTER_CTX, tpos + fl - 1 + BIAS_SEQ_OUTER_CTX)
    left_seq  = fragseq[1:BIAS_SEQ_OUTER_CTX+BIAS_SEQ_INNER_CTX]
    right_seq = fragseq[end-(BIAS_SEQ_OUTER_CTX+BIAS_SEQ_INNER_CTX)+1:end]

    a_freqs = zeros(Float64, BIAS_NUM_FREQ_BINS)
    c_freqs = zeros(Float64, BIAS_NUM_FREQ_BINS)
    g_freqs = zeros(Float64, BIAS_NUM_FREQ_BINS)
    t_freqs = zeros(Float64, BIAS_NUM_FREQ_BINS)

    fragseq = fragseq[BIAS_SEQ_OUTER_CTX+1:end-BIAS_SEQ_OUTER_CTX]
    gc = 0
    binsize = div(length(fragseq), BIAS_NUM_FREQ_BINS)
    for bin in 1:BIAS_NUM_FREQ_BINS
        if bin == BIAS_NUM_FREQ_BINS
            from = 1 + (bin - 1) * binsize
            to = length(fragseq)
        else
            from = 1 + (bin - 1) * binsize
            to = bin * binsize
        end

        for i in from:to
            if fragseq[i] == DNA_A
                a_freqs[bin] += 1
            elseif fragseq[i] == DNA_C
                gc += 1
                c_freqs[bin] += 1
            elseif fragseq[i] == DNA_G
                gc += 1
                g_freqs[bin] += 1
            elseif fragseq[i] == DNA_T
                t_freqs[bin] += 1
            end
        end

        a_freqs[bin] ./= length(from:to)
        c_freqs[bin] ./= length(from:to)
        g_freqs[bin] ./= length(from:to)
        t_freqs[bin] ./= length(from:to)
    end
    gc /= length(fragseq)

    tend = length(tseq) - BIAS_SEQ_OUTER_CTX
    # tpdist = (length(tseq) - (tpos + fl - 1)) / length(tseq)
    tpdist = length(tseq) - tpos

    return BiasTrainingExample(
        left_seq, right_seq, gc, tpdist, length(tseq), fl,
        NTuple{BIAS_NUM_FREQ_BINS, Float64}(a_freqs),
        NTuple{BIAS_NUM_FREQ_BINS, Float64}(c_freqs),
        NTuple{BIAS_NUM_FREQ_BINS, Float64}(g_freqs),
        NTuple{BIAS_NUM_FREQ_BINS, Float64}(t_freqs))
end


"""
Essentially do tseq[first:last], but pad the sequence with Ns when first:last
is out of bounds.
"""
function extract_padded_seq(tseq, first, last)
    leftpad = first < 1 ? 1 - first : 0
    rightpad = last > length(tseq) ? last - length(tseq) : 0
    fragseq = tseq[max(1, first):min(length(tseq), last)]
    if leftpad == 0 && rightpad == 0
        return fragseq
    else
        return DNASequence(repeat(dna"N", leftpad), fragseq, repeat(dna"N", rightpad))
    end
end



type BiasModel
    upctx::Int
    downctx::Int
    fs_foreground::Vector{DNASequence}
    fs_background::Vector{DNASequence}
    ss_foreground::Vector{DNASequence}
    ss_background::Vector{DNASequence}

    fs_model
    ss_model
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

    strand = ifelse(aln.flag & SAM.FLAG_REVERSE == 0, STRAND_POS, STRAND_NEG)
    pos = genomic_to_transcriptomic(t,
                ifelse(strand == STRAND_POS, aln.leftpos, aln.rightpos))
    if pos < 1
        return false, false
    end

    # adjust position in the presense of soft-clipping
    # TODO: doesn't leftposition already account for soft-clipping?
    if cigar_len(aln) > 1
        leading_clip, trailing_clip = 0, 0
        for (i, c) in enumerate(CigarIter(reads, aln))
            if i == 1 && c.op == OP_SOFT_CLIP
                leading_clip = length(c)
            else
                if c.op == OP_SOFT_CLIP
                    trailing_clip = length(c)
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


function train_model(foreground, background, k)
    # TF = TensorFlow

    # sess = TF.Session(TF.Graph())


    #=
    n0, n1 = length(background), length(foreground)
    n = n0 + n1

    ntencode = Dict{DNANucleotide, Int}(
        DNA_A => 1, DNA_C => 2, DNA_G => 3, DNA_T => 4,
        DNA_N => 1) # encode N as A, which is dumb, but it doesn't matter

    examples = zeros(Float32, (n, 4*k))
    labels = zeros(Float32, (n, 2))
    for (i, example) in enumerate(vcat(foreground, background))
        for (j, nt) in enumerate(example)
            examples[i, 4*(j-1) + ntencode[nt]] = 1
        end
        labels[i, ifelse(i <= n0, 1, 2)] = 1
    end

    x = TensorFlow.placeholder(Float32, shape=[n, 4*k])
    y_ = TensorFlow.placeholder(Float32, shape=[n, 2])
    W = TF.Variable(zeros(Float32, 4*k, 2))
    b = TF.Variable(zeros(Float32, 2))
    y = TF.nn.softmax(x*W+b)
    run(sess, TF.initialize_all_variables())

    cross_entropy = TF.reduce_mean(
        -TF.reduce_sum(y_ .* log(y), reduction_indices=[2]))

    train_step = TF.train.minimize(TF.train.GradientDescentOptimizer(.00001),
                                   cross_entropy)

    run(sess, train_step, Dict(x=>examples, y_=>labels))
    =#

    # TODO:
    #   - how do I get a weight out of this given a sequence context
    #     It's easy to do xW+b, do I just take the ratio of the two values then?
    #
    #   - how should I actually evaluate the model? Ultimately I want to improve
    #     accuracy on various benchmarks, but I can't do that at this point.
    #
    # I actually think I can score this the same way seqbias does: penalized
    # likelihood. Once I get that in place, I can start messing around with
    # models. I shouldn't go too far though. We need to build up the
    # quantification pipeline to do the ultimate evaluation.
    #
    # So let's get some basic stuff working, then focus on evaluating
    # conditional probabilities and dumping them to a sparse matrix.
    # return sess
end


function bias(bm::BiasModel, t::Transcript)
    # TODO: compute all-subsequence bias across transcript
end


function write_statistics(out::IO, bm::BiasModel)
    println(out, "strand,state,position,nucleotide,frequency")
    example_collections = [
        ("first-strand", "foreground", bm.fs_foreground),
        ("first-strand", "background", bm.fs_background),
        ("second-strand", "foreground", bm.ss_foreground),
        ("second-strand", "background", bm.ss_background)]
    freqs = Dict{DNA, Float64}(
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

