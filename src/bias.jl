

"""
One training example for bias models.
"""
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
    @assert 0.0 <= gc <= 1.0

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


struct SeqBiasModel
    orders::Vector{Int}
    ps::Array{Float32, 4}
end


# some helper functions for training seqbiasmodel


function nt2bit(nt::DNA)
    c = nt == DNA_A ? UInt8(0) :
        nt == DNA_C ? UInt8(1) :
        nt == DNA_G ? UInt8(2) :
        nt == DNA_T ? UInt8(3) :
        rand(UInt8(0):UInt8(3))
    return c
end


# update position j in the markov chain parameters ps
function seqbias_update_param_estimate!(ps, seq_train, ys_train, j, order)
    ps[j,1:end,1:end,1:end] = 1 # pseudocount

    for i in 1:size(seq_train, 1) # 0.22 seconds
        ctx = 0
        for l in 1:order
            ctx = (ctx << 2) | seq_train[i, j+l]
        end
        ps[j, ys_train[i]+1, seq_train[i, j]+1, ctx+1] += 1
    end

    for i2 in 1:2 # 0.0001 seconds
        for ctx in 1:(4^order)
            z = sum(ps[j, i2, 1:end, ctx])
            ps[j, i2, 1:end, ctx] ./= z
            for l in 1:4
                ps[j, i2, l, ctx] = log(ps[j, i2, l, ctx])
            end
        end
    end
end


function seqbias_update_lp!(ps, seq_test, seq_test_lp, j, order)
    if order < 0
        return
    end

    for i in 1:size(seq_test, 1)
        ctx = 0
        for l in 1:order
            ctx = (ctx << 2) | seq_test[i, j+l]
        end

        nt = seq_test[i, j]+1
        seq_test_lp[i, 1, j] = ps[j, 1, nt, ctx+1]
        seq_test_lp[i, 2, j] = ps[j, 2, nt, ctx+1]
    end
end


function seqbias_save_state!(ps, tmp_ps, seq_test_lp, tmp_seq_test_lp, j)
    for i1 in 1:2, i2 in 1:4, i3 in 1:size(ps, 4)
        tmp_ps[i1, i2, i3] = ps[j, i1, i2, i3]
    end

    for i1 in 1:size(seq_test_lp, 1), i2 in 1:2
        tmp_seq_test_lp[i1, i2] = seq_test_lp[i1, i2, j]
    end
end

function seqbias_restore_state!(ps, tmp_ps, seq_test_lp, tmp_seq_test_lp, j)
    for i1 in 1:2, i2 in 1:4, i3 in 1:size(ps, 4)
        ps[j, i1, i2, i3] = tmp_ps[i1, i2, i3]
    end

    for i1 in 1:size(seq_test_lp, 1), i2 in 1:2
        seq_test_lp[i1, i2, j] = tmp_seq_test_lp[i1, i2]
    end
end


function SeqBiasModel(
        foreground_training_examples::Vector{BiasTrainingExample},
        background_training_examples::Vector{BiasTrainingExample},
        foreground_testing_examples::Vector{BiasTrainingExample},
        background_testing_examples::Vector{BiasTrainingExample},
        fragend::Symbol)

    @assert fragend == :left || fragend == :right
    maxorder = 6
    maxorder_size = 4^maxorder
    k = length(foreground_training_examples[1].left_seq)

    # 2bit encode every sequence
    n_train   = length(foreground_training_examples) + length(background_training_examples)
    seq_train = Array{UInt8}((n_train, k))
    ys_train  = Array{Bool}(n_train)

    n_test    = length(foreground_testing_examples) + length(background_testing_examples)
    seq_test  = Array{UInt8}((n_test, k))
    ys_test   = Array{Bool}(n_test)

    for (seqarr, ys, offset, y, examples) in
            [(seq_train, ys_train,                                    0,  true, foreground_training_examples),
             (seq_train, ys_train, length(foreground_training_examples), false, background_training_examples),
             (seq_test,  ys_test,                                     0,  true, foreground_testing_examples),
             (seq_test,  ys_test,   length(foreground_testing_examples), false, background_testing_examples)]
        for (i, example) in enumerate(examples)
            seq = fragend == :left ? example.left_seq : example.right_seq
            for j in 1:k
                c = nt2bit(seq[j])
                seqarr[offset + i, j] = c
                ys[offset + i] = y
            end
        end
    end

    prior = length(foreground_training_examples) /
        (length(foreground_training_examples) + length(background_training_examples))

    log_prior = log(prior)
    log_inv_prior = log(1.0 - prior)

    # markov chain paremeters: indexed by
    #     position, foreground/background, nucleotide, nucleotide context
    ps = zeros(Float32, (k, 2, 4, maxorder_size))

    # arrays for saving a slice of ps
    tmp_ps = Array{Float32}((2, 4, maxorder_size))

    # probability terms, precomputed to avoid revaluating things
    seq_test_lp = zeros(Float32, (n_test, 2, k))

    # arrays for saving slices of seq_fg_test_lp/seq_bg_test_lp
    tmp_seq_test_lp = zeros(Float32, (n_test, 2))

    # order of the markov chain at each position, where -1 indicated an
    # excluded position
    orders = fill(-1, k)

    function compute_accuracy()
        lps = sum(seq_test_lp, 3)
        return mean((lps[1:end, 2] .> lps[1:end, 1]) .== ys_test)
    end

    function compute_cross_entropy()
        logcondprobs = sum(seq_test_lp, 3)
        condprobs = exp.(logcondprobs)
        logprobs = logcondprobs .- log.(condprobs[1:end, 1] .+ condprobs[1:end, 2])
        entropy = 0.0
        for i1 in 1:size(logprobs, 1)
            entropy -= ys_test[i1] * logprobs[i1, 2] + (1 - ys_test[i1]) * logprobs[i1, 1]
        end
        entropy /= size(logprobs, 1)
        return entropy
    end

    accuracy0 = compute_accuracy()
    entropy0 = compute_cross_entropy()
    while true
        best_accuracy = accuracy0
        best_entropy = entropy0
        best_j = 0

        for j in 1:k
            if orders[j] < maxorder && j + orders[j] < k
                # save parameters
                seqbias_save_state!(ps, tmp_ps, seq_test_lp, tmp_seq_test_lp, j)

                # increase order and try out the model
                orders[j] += 1

                seqbias_update_param_estimate!(
                    ps, seq_train, ys_train, j, orders[j]) # 0.0005 seconds
                seqbias_update_lp!(
                    ps, seq_test, seq_test_lp, j, orders[j]) # 0.0001 seconds

                accuracy = compute_accuracy() # 0.001 seconds
                entropy = compute_cross_entropy() # 0.001 seconds

                # @show (j, orders[j], accuracy)
                # if accuracy > best_accuracy
                if entropy < best_entropy
                    best_accuracy = accuracy
                    best_entropy = entropy
                    best_j = j
                end

                # return parameters to original state
                orders[j] -= 1
                seqbias_restore_state!(ps, tmp_ps, seq_test_lp, tmp_seq_test_lp, j)
            end
        end

        # if best_accuracy <= accuracy0
        if best_entropy >= entropy0
            break
        else
            orders[best_j] += 1
            seqbias_update_param_estimate!(
                ps, seq_train, ys_train, best_j, orders[best_j])
            seqbias_update_lp!(ps, seq_test, seq_test_lp, best_j, orders[best_j])
            accuracy0 = best_accuracy
            entropy0 = best_entropy
        end
    end

    @show accuracy0
    @show entropy0
    @show orders

    # TODO: I can make this faster by storing foreground/background ratios in ps
    # instead of having two separate arrays

    return SeqBiasModel(orders, ps)
end


function evaluate(sb::SeqBiasModel, seq::DNASequence)
    @assert length(seq) == length(sb.orders)
    bias = 0.0
    for i in 1:length(seq)
        if sb.orders[i] > 0
            c = nt2bit(seq[i])
            ctx = 0
            for l in 1:sb.orders[i]
                ctx = (ctx << 2) | nt2bit(seq[i+l])
            end
            bias += sb.ps[i, 2, c+1, ctx+1] - sb.ps[i, 1, c+1, ctx+1]
        end
    end
    return bias
end


struct SimpleHistogramModel
    qs::Vector{Float32} # quantiles
    bins::Vector{Float32} # log-probability ratios
end


function SimpleHistogramModel(
        xs::Vector{Float32}, ys::Vector{Bool}, weights::Vector{Float32})
    numbins = 40
    total_weight = sum(weights)
    binsize = total_weight/numbins

    p = sortperm(xs)
    xs      = xs[p]
    weights = weights[p]
    ys      = ys[p]

    # define bin quantiles
    qs = Vector{Float32}(numbins-1)
    nextbin = 1
    wsum = 0.0
    for (x, w) in zip(xs, weights)
        wsum += w
        if wsum > nextbin*binsize
            qs[nextbin] = x
            nextbin += 1
            if nextbin == numbins
                break
            end
        end
    end
    @show qs

    bincounts = ones(Float32, (2, numbins))
    for (x, y, w) in zip(xs, ys, weights)
        i = searchsorted(qs, x).start
        bincounts[y+1, i] += w
    end

    bincounts_sums = sum(bincounts, 2)
    @show bincounts
    @show bincounts_sums
    bins = Vector{Float32}(numbins)
    for i in 1:numbins
        bins[i] =
            log(bincounts[2, i]/bincounts_sums[2]) -
            log(bincounts[1, i]/bincounts_sums[1])
    end
    @show bins

    return SimpleHistogramModel(qs, bins)
end


function evaluate(hist::SimpleHistogramModel, x)
    i = searchsorted(hist.qs, x).start
    return hist.bins[i]
end


struct BiasModel
    left_seqbias::SeqBiasModel
    right_seqbias::SeqBiasModel
    gc_model::SimpleHistogramModel
end


"""
Train ensemble bias model.
"""
function BiasModel(
        bias_foreground_training_examples::Vector{BiasTrainingExample},
        bias_background_training_examples::Vector{BiasTrainingExample},
        bias_foreground_testing_examples::Vector{BiasTrainingExample},
        bias_background_testing_examples::Vector{BiasTrainingExample})

    # train sequence bias models
    @time seqbias_left = SeqBiasModel(
        bias_foreground_training_examples,
        bias_background_training_examples,
        bias_foreground_testing_examples,
        bias_background_testing_examples,
        :left)

    @time seqbias_right = SeqBiasModel(
        bias_foreground_training_examples,
        bias_background_training_examples,
        bias_foreground_testing_examples,
        bias_background_testing_examples,
        :right)

    # TODO: weight training examples

    # collect GC information
    n_training =
        length(bias_foreground_training_examples) +
        length(bias_background_training_examples)
    frag_gc_training = Vector{Float32}(n_training)
    ys_training = Vector{Bool}(n_training)
    for (i, example) in enumerate(bias_foreground_training_examples)
        frag_gc_training[i] = example.frag_gc
        ys_training[i] = true
    end
    for (i, example) in enumerate(bias_background_training_examples)
        idx = length(bias_foreground_training_examples) + i
        frag_gc_training[idx] = example.frag_gc
        ys_training[idx] = false
    end

    n_testing =
        length(bias_foreground_testing_examples) +
        length(bias_background_testing_examples)
    frag_gc_testing = Vector{Float32}(n_testing)
    ys_testing = Vector{Bool}(n_testing)
    for (i, example) in enumerate(bias_foreground_testing_examples)
        frag_gc_testing[i] = example.frag_gc
        ys_testing[i] = true
    end
    for (i, example) in enumerate(bias_background_testing_examples)
        idx = length(bias_foreground_testing_examples) + i
        frag_gc_testing[idx] = example.frag_gc
        ys_testing[idx] = false
    end

    gc_model = SimpleHistogramModel(
        frag_gc_training, ys_training, ones(Float32, n_training))

    bm = BiasModel(seqbias_left, seqbias_right, gc_model)

    @show accuracy(
        bm, bias_foreground_testing_examples, bias_background_testing_examples)

    return bm
end



function accuracy(
        bm::BiasModel, foreground_testing_examples, background_testing_examples)
    acc = 0
    for example in foreground_testing_examples
        bias =
            evaluate(bm.left_seqbias, example.left_seq) +
            evaluate(bm.right_seqbias, example.right_seq) +
            evaluate(bm.gc_model, example.frag_gc)
        acc += bias > 0.0
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.left_seqbias, example.left_seq) +
            evaluate(bm.right_seqbias, example.right_seq) +
            evaluate(bm.gc_model, example.frag_gc)
        acc += bias < 0.0
    end

    return acc / (length(foreground_testing_examples) + length(background_testing_examples))
end
