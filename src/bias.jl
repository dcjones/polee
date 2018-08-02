

"""
One training example for bias models.
"""
struct BiasTrainingExample
    left_seq::Vector{DNA}
    right_seq::Vector{DNA}
    frag_gc::Float64
    tpdist::Int
    fpdist::Int
    tlen::Int
    flen::Int
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

    # tpdist = (length(tseq) - (tpos + fl - 1)) / length(tseq)
    tpdist = length(tseq) - (tpos + fl - 1)
    # tpdist = length(tseq) - tpos
    # fpdist = tpos

    # distance of the 5' end of the fragment from the 3' end of the transcript
    fpdist = length(tseq) - tpos + 1

    return BiasTrainingExample(
        left_seq, right_seq, gc, tpdist, fpdist, length(tseq), fl,
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
        return vcat(fill(DNA_N, leftpad), fragseq, fill(DNA_N, rightpad))
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
    ps::Array{Float32, 3}
end


# some helper functions for training seqbiasmodel


# function nt2bit(nt::DNA)
#     c = nt == DNA_A ? UInt8(0) :
#         nt == DNA_C ? UInt8(1) :
#         nt == DNA_G ? UInt8(2) :
#         nt == DNA_T ? UInt8(3) :
#         rand(UInt8(0):UInt8(3))
#     return c
# end


const nt2bit = UInt8[
    0,                   # A
    1, 0,                # C
    2, 0, 0, 0,          # C
    3, 0, 0, 0, 0, 0, 0, # G
    0 ]                  # N

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
        end
    end
end


function seqbias_update_p!(ps, seq_test, seq_test_p, j, order)
    if order < 0
        return
    end

    for i in 1:size(seq_test, 1)
        ctx = 0
        for l in 1:order
            ctx = (ctx << 2) | seq_test[i, j+l]
        end

        nt = seq_test[i, j]+1
        seq_test_p[i, 1, j] = ps[j, 1, nt, ctx+1]
        seq_test_p[i, 2, j] = ps[j, 2, nt, ctx+1]
    end
end


function seqbias_save_state!(ps, tmp_ps, seq_test_p, tmp_seq_test_p, j)
    for i1 in 1:2, i2 in 1:4, i3 in 1:size(ps, 4)
        tmp_ps[i1, i2, i3] = ps[j, i1, i2, i3]
    end

    for i1 in 1:size(seq_test_p, 1), i2 in 1:2
        tmp_seq_test_p[i1, i2] = seq_test_p[i1, i2, j]
    end
end

function seqbias_restore_state!(ps, tmp_ps, seq_test_p, tmp_seq_test_p, j)
    for i1 in 1:2, i2 in 1:4, i3 in 1:size(ps, 4)
        ps[j, i1, i2, i3] = tmp_ps[i1, i2, i3]
    end

    for i1 in 1:size(seq_test_p, 1), i2 in 1:2
        seq_test_p[i1, i2, j] = tmp_seq_test_p[i1, i2]
    end
end


function compute_seqbias_loss(seq_test_p, ys_test)
    n_test, _, k = size(seq_test_p)
    loss = 0.0
    for i in 1:n_test
        p_fg = 1.0
        p_bg = 1.0
        for j in 1:k
            p_fg *= seq_test_p[i, 2, j]
            p_bg *= seq_test_p[i, 1, j]
        end
        p = p_fg / (p_fg + p_bg)
        loss += ifelse(ys_test[i], 1.0 - p, p)
    end
    return loss
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
                c = nt2bit[Int(seq[j])]
                seqarr[offset + i, j] = c
                ys[offset + i] = y
            end
        end
    end

    prior = length(foreground_training_examples) /
        (length(foreground_training_examples) + length(background_training_examples))

    # markov chain paremeters: indexed by
    #     position, foreground/background, nucleotide, nucleotide context
    ps = zeros(Float32, (k, 2, 4, maxorder_size))

    # arrays for saving a slice of ps
    tmp_ps = Array{Float32}((2, 4, maxorder_size))

    # probability terms, precomputed to avoid revaluating things
    seq_test_p = ones(Float32, (n_test, 2, k))

    # arrays for saving slices of seq_fg_test_lp/seq_bg_test_lp
    tmp_seq_test_p = ones(Float32, (n_test, 2))

    # order of the markov chain at each position, where -1 indicated an
    # excluded position
    orders = fill(-1, k)

    loss0 = compute_seqbias_loss(seq_test_p, ys_test)
    while true
        best_loss = loss0
        best_j = 0

        for j in 1:k
            if orders[j] < maxorder && j + orders[j] < k
                # save parameters
                seqbias_save_state!(ps, tmp_ps, seq_test_p, tmp_seq_test_p, j)

                # increase order and try out the model
                orders[j] += 1

                seqbias_update_param_estimate!(
                    ps, seq_train, ys_train, j, orders[j]) # 0.0005 seconds
                seqbias_update_p!(
                    ps, seq_test, seq_test_p, j, orders[j]) # 0.0001 seconds

                loss = compute_seqbias_loss(seq_test_p, ys_test)

                # if accuracy > best_accuracy
                if loss < best_loss
                    best_loss = loss
                    best_j = j
                end

                # return parameters to original state
                orders[j] -= 1
                seqbias_restore_state!(ps, tmp_ps, seq_test_p, tmp_seq_test_p, j)
            end
        end

        if best_loss >= loss0
            break
        else
            orders[best_j] += 1
            seqbias_update_param_estimate!(
                ps, seq_train, ys_train, best_j, orders[best_j])
            seqbias_update_p!(ps, seq_test, seq_test_p, best_j, orders[best_j])
            loss0 = best_loss
        end
    end

    return SeqBiasModel(orders, ps[:,2,:,:] ./ ps[:,1,:,:])
end


function evaluate(sb::SeqBiasModel, seq)
    @assert length(seq) == length(sb.orders)
    bias = 1.0f0
    @inbounds for i in 1:length(seq)
        if sb.orders[i] > 0
            c = nt2bit[Int(seq[i])]
            ctx = 0
            for l in 1:sb.orders[i]
                ctx = (ctx << 2) | nt2bit[Int(seq[i+l])]
            end
            bias *= sb.ps[i, c+1, ctx+1]
        end
    end
    return bias
end


function evaluate(sb::SeqBiasModel, seq, pos)
    bias = 1.0f0
    seqlen = length(seq)
    @inbounds for (i, j) in enumerate(pos-BIAS_SEQ_OUTER_CTX:pos+BIAS_SEQ_INNER_CTX-1)
        if sb.orders[i] > 0
            c = nt2bit[Int(1 <= j <= seqlen ? seq[j] : DNA_N)]
            ctx = 0
            for l in 1:sb.orders[i]
                ctx = (ctx << 2) | nt2bit[Int(1 <= j+l <= seqlen ? seq[j+l] : DNA_N)]
            end
            bias *= sb.ps[i, c+1, ctx+1]
        end
    end

    return bias
end

# TODO: I'm afraid the only real solution is to 2-bit encode the sequences so
# we can avoid the overhead of of decoding. I don't really understand why this
# is so slow though.


struct SimpleHistogramModel
    bins::Vector{Float32} # probability ratios
end


function SimpleHistogramModel(
        xs::Vector{Float32}, ys::Vector{Bool}, weights::Vector{Float32})
    total_weight = sum(weights)

    p = sortperm(xs)
    xs      = xs[p]
    weights = weights[p]
    ys      = ys[p]

    numbins = 15
    binsize = total_weight/numbins

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

    bincounts = ones(Float32, (2, numbins))
    for (x, y, w) in zip(xs, ys, weights)
        i = searchsorted(qs, x).start
        bincounts[y+1, i] += w
    end

    bincounts_sums = sum(bincounts, 2)
    bins = Vector{Float32}(numbins)
    for i in 1:numbins
        bins[i] =
            (bincounts[2, i]/bincounts_sums[2]) /
            (bincounts[1, i]/bincounts_sums[1])
    end

    # expand bins into finer grained uniformly spaced bins for faster lookup
    expanded_bins = Vector{Float32}(100)
    for i in 1:length(expanded_bins)
        q = (i - 0.5) / length(expanded_bins)
        j = searchsorted(qs, q).start
        expanded_bins[i] = bins[j]
    end

    return SimpleHistogramModel(expanded_bins)
end


function evaluate(hist::SimpleHistogramModel, x)
    i = clamp(round(Int, x * length(hist.bins)), 1, length(hist.bins))
    return hist.bins[i]
end


struct PositionalBiasModel
    p::Float64
    terms::Vector{Float64}
end


"""
Fit geometric 3' bias model by gradient ascent on the likelihood.
"""
function fit_pos_bias(tlens, fpdists, flens, maxtlen, efflens, fraglen_pmf)
    logprob_terms = Vector{Float64}(maxtlen)
    logprob_grad_terms = Vector{Float64}(maxtlen)

    adam_rm = 0.9
    adam_rv = 0.9
    adam_m = 0.0
    adam_v = 0.0
    adam_eps = 1e-10

    logit_p = logit(1e-10)
    for it in 1:1000
        step_size = 0.1 * exp(-5e-3 * it)
        p = logistic(logit_p)

        logprob_terms[1] = 0.0
        logprob_grad_terms[1] = 0.0
        for k in 1:maxtlen-1
            logprob_terms[k+1] = (1/efflens[k]) * p * (1-p)^k
            logprob_grad_terms[k+1] = - (1/efflens[k]) * (1 - p)^(k-1) * (k*p + p - 1)
        end
        cumsum!(logprob_terms, logprob_terms)
        cumsum!(logprob_grad_terms, logprob_grad_terms)

        lp = 0.0
        lp_grad = 0.0
        for (fpdist, tlen) in zip(fpdists, tlens)
            term = logprob_terms[tlen] - logprob_terms[fpdist]
            prob = term + (1/efflens[tlen]) * (1-p)^tlen
            lp += log(prob)

            term_grad = logprob_grad_terms[tlen] - logprob_grad_terms[fpdist]
            prob_grad = term_grad - (1/efflens[tlen]) * tlen * (1-p)^(tlen - 1)
            lp_grad += (1/prob) * prob_grad
        end

        dlogisticp_dp = p * logistic(-logit_p)
        lp_logit_p_grad = lp_grad * dlogisticp_dp

        # adam update
        if it == 1
            adam_m = lp_logit_p_grad
            adam_v = lp_logit_p_grad^2
        else
            adam_m = adam_rm * adam_m + (1 - adam_rm) * lp_logit_p_grad
            adam_v = adam_rv * adam_v + (1 - adam_rv) * lp_logit_p_grad^2
        end

        m_denom = (1 - adam_rm^it)
        v_denom = (1 - adam_rv^it)
        param_m = adam_m / m_denom
        param_v = adam_v / v_denom
        delta = step_size * param_m / (sqrt(param_v) + adam_eps)

        logit_p += delta
    end

    # recompute terms without fragment length adjustment
    p = logistic(logit_p)
    for k in 1:maxtlen-1
        logprob_terms[k+1] = (1/k) * p * (1-p)^k
    end
    cumsum!(logprob_terms, logprob_terms)

    return p, logprob_terms
end


function PositionalBiasModel(
        ts::Transcripts, fraglen_pmf::Vector{Float32},
        foreground_training_examples::Vector{BiasTrainingExample},
        background_training_examples::Vector{BiasTrainingExample},
        foreground_testing_examples::Vector{BiasTrainingExample},
        background_testing_examples::Vector{BiasTrainingExample},
        weights::Vector{Float32})

    maxtlen = 0
    for t in ts
        maxtlen = max(maxtlen, exonic_length(t))
    end

    # copy relavent data from examples to flat arrays
    poss_fg_train  = Vector{Int}(length(foreground_training_examples))
    tlens_fg_train = Vector{Int}(length(foreground_training_examples))
    flens_fg_train = Vector{Int}(length(foreground_training_examples))

    for (i, example) in enumerate(foreground_training_examples)
        poss_fg_train[i]  = example.fpdist
        tlens_fg_train[i] = example.tlen
        flens_fg_train[i] = example.flen
    end

    # need effective lengths to compute positional bias accounting for
    # fragment lengths
    efflens = Vector{Float64}(maxtlen)
    for tlen in 1:maxtlen
        efflen = 0.0
        for (flen, flen_pr) in enumerate(fraglen_pmf)
            if flen > tlen
                break
            end
            efflen += flen_pr * (tlen - flen + 1)
        end
        efflens[tlen] = efflen
    end

    p, terms = fit_pos_bias(
        tlens_fg_train, poss_fg_train, flens_fg_train,
        maxtlen, efflens, fraglen_pmf)

    println("Positional bias rate: ", p)

    return PositionalBiasModel(p, terms)
end


function evaluate(posmodel::PositionalBiasModel, tlen, pos; classification::Bool=false)
    prob = (1/tlen) * (1-posmodel.p)^tlen + posmodel.terms[tlen] - posmodel.terms[pos]
    if classification
        return tlen * prob
    else
        # adjust so the 3' end has bias 1.0.
        c = (1/tlen) * (1-posmodel.p)^tlen + posmodel.terms[tlen]
        return prob / c
    end
end


struct BiasModel
    left_seqbias::SeqBiasModel
    right_seqbias::SeqBiasModel
    gc_model::SimpleHistogramModel
    pos_model::PositionalBiasModel
end


"""
Train ensemble bias model.
"""
function BiasModel(
        ts::Transcripts, fraglen_pmf::Vector{Float32},
        bias_foreground_training_examples::Vector{BiasTrainingExample},
        bias_background_training_examples::Vector{BiasTrainingExample},
        bias_foreground_testing_examples::Vector{BiasTrainingExample},
        bias_background_testing_examples::Vector{BiasTrainingExample})

    n_training =
        length(bias_foreground_training_examples) +
        length(bias_background_training_examples)

    n_testing =
        length(bias_foreground_testing_examples) +
        length(bias_background_testing_examples)

    # train sequence bias models
    # --------------------------

    println("Fitting sequence bias model...")

    seqbias_left = SeqBiasModel(
        bias_foreground_training_examples,
        bias_background_training_examples,
        bias_foreground_testing_examples,
        bias_background_testing_examples,
        :left)

    seqbias_right = SeqBiasModel(
        bias_foreground_training_examples,
        bias_background_training_examples,
        bias_foreground_testing_examples,
        bias_background_testing_examples,
        :right)

    # weight training examples
    # ------------------------

    weights = Vector{Float32}(n_training)
    bs = Float32[]

    for (i, example) in enumerate(bias_foreground_training_examples)
        b = evaluate(seqbias_left, example.left_seq) +
            evaluate(seqbias_right, example.right_seq)
        push!(bs, b)
        weights[i] = 1.0 - logistic(b)
    end
    for (i, example) in enumerate(bias_foreground_training_examples)
        b = evaluate(seqbias_left, example.left_seq) +
            evaluate(seqbias_right, example.right_seq)
        push!(bs, b)
        idx = length(bias_foreground_training_examples) + i
        weights[idx] = logistic(b)
    end

    # train fragment GC model
    # -----------------------

    # collect GC information
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

    println("Fitting GC content bias model...")

    fill!(weights, 1.0f0) # seems to do better without weigths
    gc_model = SimpleHistogramModel(
        frag_gc_training, ys_training, weights)

    # re-weight training examples
    # ---------------------------

    for (i, example) in enumerate(bias_foreground_training_examples)
        weights[i] *= 1.0 - logistic(evaluate(gc_model, example.frag_gc))
    end

    for (i, example) in enumerate(bias_foreground_training_examples)
        idx = length(bias_foreground_training_examples) + i
        weights[idx] *= logistic(evaluate(gc_model, example.frag_gc))
    end

    # train positional bias model
    # ---------------------------

    println("Fitting positional bias model...")

    fill!(weights, 1.0f0) # seems to do better without weigths
    pos_model = PositionalBiasModel(
        ts, fraglen_pmf,
        bias_foreground_training_examples,
        bias_background_training_examples,
        bias_foreground_testing_examples,
        bias_background_testing_examples,
        weights)

    bm = BiasModel(seqbias_left, seqbias_right, gc_model, pos_model)

    for (i, example) in enumerate(bias_foreground_training_examples)
        weights[i] = evaluate(bm.pos_model, example.tlen, example.fpdist)
    end

    for (i, example) in enumerate(bias_foreground_training_examples)
        idx = length(bias_foreground_training_examples) + i
        weights[idx] = evaluate(bm.pos_model, example.tlen, example.fpdist)
    end
    @show median(weights)
    @show extrema(weights)

    for (i, example) in enumerate(bias_foreground_training_examples)
        weights[i] = evaluate(bm.gc_model, example.frag_gc)
    end

    for (i, example) in enumerate(bias_foreground_training_examples)
        idx = length(bias_foreground_training_examples) + i
        weights[idx] = evaluate(bm.gc_model, example.frag_gc)
    end
    @show median(weights)
    @show extrema(weights)

    @show accuracy1(bm, bias_foreground_testing_examples, bias_background_testing_examples)
    @show accuracy2(bm, bias_foreground_testing_examples, bias_background_testing_examples)
    @show accuracy3(bm, bias_foreground_testing_examples, bias_background_testing_examples)
    @show accuracy4(bm, bias_foreground_testing_examples, bias_background_testing_examples)

    acc = accuracy(
        bm, bias_foreground_testing_examples, bias_background_testing_examples)
    @printf("Bias model accuracy: %0.2f%%\n", acc)

    return bm
end


function accuracy(
        bm::BiasModel, foreground_testing_examples, background_testing_examples)
    acc = 0

    bs = Vector{Float64}(length(foreground_testing_examples) + length(background_testing_examples))
    idx = 1
    for example in foreground_testing_examples
        bias =
            evaluate(bm.left_seqbias, example.left_seq) *
            evaluate(bm.right_seqbias, example.right_seq) *
            evaluate(bm.gc_model, example.frag_gc) *
            evaluate(bm.pos_model, example.tlen, example.fpdist, classification=true)
        bs[idx] = bias
        idx += 1
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.left_seqbias, example.left_seq) *
            evaluate(bm.right_seqbias, example.right_seq) *
            evaluate(bm.gc_model, example.frag_gc) *
            evaluate(bm.pos_model, example.tlen, example.fpdist, classification=true)
        bs[idx] = bias
        idx += 1
    end

    bs .-= median(bs)

    for i in 1:length(foreground_testing_examples)
        acc += bs[i] .> 0.0
    end

    for i in 1:length(background_testing_examples)
        acc += bs[length(foreground_testing_examples) + i] .<= 0.0
    end

    return acc / (length(foreground_testing_examples) + length(background_testing_examples))
end


function accuracy1(
        bm::BiasModel, foreground_testing_examples, background_testing_examples)
    acc = 0

    bs = Vector{Float64}(length(foreground_testing_examples) + length(background_testing_examples))
    idx = 1
    for example in foreground_testing_examples
        bias =
            evaluate(bm.left_seqbias, example.left_seq)
        bs[idx] = bias
        idx += 1
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.left_seqbias, example.left_seq)
        bs[idx] = bias
        idx += 1
    end

    bs .-= median(bs)

    for i in 1:length(foreground_testing_examples)
        acc += bs[i] .> 0.0
    end

    for i in 1:length(background_testing_examples)
        acc += bs[length(foreground_testing_examples) + i] .<= 0.0
    end

    return acc / (length(foreground_testing_examples) + length(background_testing_examples))
end


function accuracy2(
        bm::BiasModel, foreground_testing_examples, background_testing_examples)
    acc = 0

    bs = Vector{Float64}(length(foreground_testing_examples) + length(background_testing_examples))
    idx = 1
    for example in foreground_testing_examples
        bias =
            evaluate(bm.right_seqbias, example.right_seq)
        bs[idx] = bias
        idx += 1
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.right_seqbias, example.right_seq)
        bs[idx] = bias
        idx += 1
    end

    bs .-= median(bs)

    for i in 1:length(foreground_testing_examples)
        acc += bs[i] .> 0.0
    end

    for i in 1:length(background_testing_examples)
        acc += bs[length(foreground_testing_examples) + i] .<= 0.0
    end

    return acc / (length(foreground_testing_examples) + length(background_testing_examples))
end


function accuracy3(
        bm::BiasModel, foreground_testing_examples, background_testing_examples)
    acc = 0

    bs = Vector{Float64}(length(foreground_testing_examples) + length(background_testing_examples))
    idx = 1
    for example in foreground_testing_examples
        bias =
            evaluate(bm.gc_model, example.frag_gc)
        bs[idx] = bias
        idx += 1
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.gc_model, example.frag_gc)
        bs[idx] = bias
        idx += 1
    end

    bs .-= median(bs)

    for i in 1:length(foreground_testing_examples)
        acc += bs[i] .> 0.0
    end

    for i in 1:length(background_testing_examples)
        acc += bs[length(foreground_testing_examples) + i] .<= 0.0
    end

    return acc / (length(foreground_testing_examples) + length(background_testing_examples))
end


function accuracy4(
        bm::BiasModel, foreground_testing_examples, background_testing_examples)
    acc = 0

    bs = Vector{Float64}(length(foreground_testing_examples) + length(background_testing_examples))
    idx = 1
    for example in foreground_testing_examples
        bias =
            evaluate(bm.pos_model, example.tlen, example.fpdist, classification=true)
        bs[idx] = bias
        idx += 1
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.pos_model, example.tlen, example.fpdist, classification=true)
        bs[idx] = bias
        idx += 1
    end

    bs .-= median(bs)

    for i in 1:length(foreground_testing_examples)
        acc += bs[i] .> 0.0
    end

    for i in 1:length(background_testing_examples)
        acc += bs[length(foreground_testing_examples) + i] .<= 0.0
    end

    return acc / (length(foreground_testing_examples) + length(background_testing_examples))
end


"""
Compute bias for fragment left and right ends.
"""
function compute_transcript_bias!(bm::BiasModel, t::Transcript)
    tseq = t.metadata.seq
    tlen = length(tseq)

    left_bias = t.metadata.left_bias
    right_bias = t.metadata.right_bias

    resize!(left_bias, tlen)
    resize!(right_bias, tlen)

    # left bias
    for pos in 1:tlen
        left_bias[pos] =
            evaluate(bm.pos_model, tlen, tlen - pos + 1) *
            evaluate(bm.left_seqbias, tseq, pos)
    end

    # right bias
    for pos in 1:tlen
        right_bias[pos] =
            evaluate(bm.right_seqbias, tseq, pos)
    end
end

