

"""
One training example for bias models.
"""
struct BiasTrainingExample
    left_seq::Vector{DNA}
    right_seq::Vector{DNA}
    frag_gc::Float64
    tpdist::Float64
    fpdist::Float64
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

    # log_prior = log(prior)
    # log_inv_prior = log(1.0 - prior)

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

    function compute_accuracy()
        lps = prod(seq_test_p, 3)
        return mean((lps[1:end, 2] .> lps[1:end, 1]) .== ys_test)
    end

    function compute_cross_entropy()
        condprobs = prod(seq_test_p, 3)
        logprobs = log.(condprobs ./ (condprobs[1:end, 1] .+ condprobs[1:end, 2]))
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
                seqbias_save_state!(ps, tmp_ps, seq_test_p, tmp_seq_test_p, j)

                # increase order and try out the model
                orders[j] += 1

                seqbias_update_param_estimate!(
                    ps, seq_train, ys_train, j, orders[j]) # 0.0005 seconds
                seqbias_update_p!(
                    ps, seq_test, seq_test_p, j, orders[j]) # 0.0001 seconds

                accuracy = compute_accuracy() # 0.001 seconds
                entropy = compute_cross_entropy() # 0.001 seconds

                # if accuracy > best_accuracy
                if entropy < best_entropy
                    best_accuracy = accuracy
                    best_entropy = entropy
                    best_j = j
                end

                # return parameters to original state
                orders[j] -= 1
                seqbias_restore_state!(ps, tmp_ps, seq_test_p, tmp_seq_test_p, j)
            end
        end

        # if best_accuracy <= accuracy0
        if best_entropy >= entropy0
            break
        else
            orders[best_j] += 1
            seqbias_update_param_estimate!(
                ps, seq_train, ys_train, best_j, orders[best_j])
            seqbias_update_p!(ps, seq_test, seq_test_p, best_j, orders[best_j])
            accuracy0 = best_accuracy
            entropy0 = best_entropy
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


const PosInterType =
    ScaledInterpolation{Float32,2,
        Interpolations.BSplineInterpolation{Float32,2,Array{Float32,2},BSpline{Linear},OnGrid,0},
        BSpline{Linear},
        OnGrid,
        Tuple{
            StepRangeLen{Float32,Base.TwicePrecision{Float32},Base.TwicePrecision{Float32}},
            StepRangeLen{Float32,Base.TwicePrecision{Float32},Base.TwicePrecision{Float32}}}}

struct PositionalBiasModel
    intp::PosInterType
    max_len::Int
end


function smooth_rows(xs)
    c = 0.1
    ys = similar(xs)
    for i in 1:size(xs, 1)
        for j in 1:size(xs, 2)
            if j == 1
                ys[i, j] =
                    (1.0 - c) * xs[i, j] +
                    c * xs[i, j+1]
            elseif j == size(xs, 2)
                ys[i, j] =
                    c * xs[i, j-1] +
                    (1.0 - c) * xs[i, j]
            else
                ys[i, j] =
                    c * xs[i, j-1] +
                    (1.0 - 2*c) * xs[i, j] +
                    c * xs[i, j+1]
            end
        end
    end
    return ys
end


function fit_interpolation(
        poss_fg, tlens_fg, poss_bg, tlens_bg,
        numlenbins, lenbinsize, numposbins;
        normalize_to_mode::Bool=false)
    lenbins = collect(lenbinsize:lenbinsize:(lenbinsize * numlenbins))

    kd_sd = 0.05

    posbins = collect(linspace(0.0, 1.0, numposbins+1))[2:end]
    posbinsize = length(posbins) > 1 ? posbins[2] - posbins[1] : 1.0

    weights_fg = ones(Float64, length(tlens_fg))
    A = fill(1.0f0, (numlenbins, numposbins))
    # for (tlen, pos, w) in zip(tlens_fg, poss_fg, weights_fg)
    #     if tlen > lenbins[end]
    #         continue
    #     end
    #     i = min(numlenbins, searchsorted(lenbins, tlen).start)
    #     j = min(numposbins, searchsorted(posbins, pos).start)
    #     A[i, j] += w
    # end
    for (tlen, pos, w) in zip(tlens_fg, poss_fg, weights_fg)
        if tlen > lenbins[end]
            continue
        end
        i = min(numlenbins, searchsorted(lenbins, tlen).start)

        d_sum = 0.0
        for j in 1:numposbins
            binpos = (j - 0.5) / numposbins
            d = exp(-(binpos - pos)^2 / (2*kd_sd))
            d_sum += d
        end

        for j in 1:numposbins
            binpos = (j - 0.5) / numposbins
            d = exp(-(binpos - pos)^2 / (2*kd_sd))
            A[i, j] += d/d_sum * w
        end
    end
    A ./= sum(A, 2)

    weights_bg = ones(Float64, length(tlens_bg))
    B = fill(1.0f0, (numlenbins, numposbins))
    # for (tlen, pos, w) in zip(tlens_bg, poss_bg, weights_bg)
    #     if tlen > lenbins[end]
    #         continue
    #     end
    #     i = min(numlenbins, searchsorted(lenbins, tlen).start)
    #     j = min(numposbins, searchsorted(posbins, pos).start)
    #     B[i, j] += w
    # end
    for (tlen, pos, w) in zip(tlens_fg, poss_bg, weights_bg)
        if tlen > lenbins[end]
            continue
        end
        i = min(numlenbins, searchsorted(lenbins, tlen).start)

        d_sum = 0.0
        for j in 1:numposbins
            binpos = (j - 0.5) / numposbins
            d = exp(-(binpos - pos)^2 / (2*kd_sd))
            d_sum += d
        end

        for j in 1:numposbins
            binpos = (j - 0.5) / numposbins
            d = exp(-(binpos - pos)^2 / (2*kd_sd))
            B[i, j] += (d/d_sum) * w
        end
    end
    B ./= sum(B, 2)

    # if normalize_to_mode
        # println("-----------------------------")
        # @show lenbins
        # @show posbins

        # for i in 1:size(A, 1)
        #     @show (i, A[i, :])
        # end

        # for i in 1:size(A, 1)
        #     @show (i, B[i, :])
        # end
    # end

    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            A[i, j] = A[i, j] / B[i, j]
        end
    end

    # normalize to mode
    if normalize_to_mode
        for i in 1:size(A, 1)
            max_bin = 0
            max_bin_prob = 0.0
            for j in 1:size(A, 2)
                if A[i, j] > max_bin_prob
                    max_bin = j
                    max_bin_prob = A[i, j]
                end
            end

            @show (i, max_bin, max_bin_prob)
            A_adj = A[i, max_bin]
            # B_adj = B[i, max_bin]
            for j in 1:size(A, 2)
                # A[i, j] = (A[i, j] / A_adj) / (B[i, j] / B_adj)
                A[i, j] = (A[i, j] / A_adj)
            end
        end
    end

    println("-----------------------------")
    @show lenbins
    @show posbins

    for i in 1:size(A, 1)
        @show (i, A[i, :])
    end


    if normalize_to_mode
        for i in 1:size(A, 1)
            @show (i, A[i, :])
        end
    end

    # itp = interpolate(A, BSpline(Constant()), OnGrid())
    itp = interpolate(A, BSpline(Linear()), OnGrid())
    # itp = interpolate(A, BSpline(Quadratic(Flat())), OnGrid())
    # itp = interpolate(A, BSpline(Cubic(Natural())), OnGrid())

    lenscale = Float32(lenbins[1] - lenbinsize/2):Float32(lenbinsize):Float32(lenbins[end] - lenbinsize/2)
    posscale = Float32(posbins[1] - posbinsize/2):Float32(posbinsize):Float32(posbins[end] - posbinsize/2)
    sitp = Interpolations.scale(itp, lenscale, posscale)

    return sitp
end


function compute_loss(
        model::PositionalBiasModel,
        poss_fg_test, tlens_fg_test,
        poss_bg_test, tlens_bg_test)
    # entropy = 0.0
    # for (tlen, pos) in zip(tlens_fg_test, poss_fg_test)
    #     p = logistic(log(evaluate(model, tlen, pos)))
    #     entropy -= log(p)
    # end

    # for (tlen, pos) in zip(tlens_bg_test, poss_bg_test)
    #     p = logistic(log(evaluate(model, tlen, pos)))
    #     entropy -= log(1 - p)
    # end

    # return entropy/(length(tlens_fg_test) + length(tlens_bg_test))

    loss = 0.0
    for (tlen, pos) in zip(tlens_fg_test, poss_fg_test)
        loss += (1.0 - 1 * (evaluate(model, tlen, pos) > 1.0 ? 1.0 : -1.0))^2
    end

    for (tlen, pos) in zip(tlens_bg_test, poss_bg_test)
        loss += (-1.0 - 1 * (evaluate(model, tlen, pos) > 1.0 ? 1.0 : -1.0))^2
    end

    return loss
end

function PositionalBiasModel(
        foreground_training_examples::Vector{BiasTrainingExample},
        background_training_examples::Vector{BiasTrainingExample},
        foreground_testing_examples::Vector{BiasTrainingExample},
        background_testing_examples::Vector{BiasTrainingExample},
        weights::Vector{Float32})

    # copy relavent data from examples to flat arrays
    poss_fg_train = Vector{Float32}(length(foreground_training_examples))
    tlens_fg_train   = Vector{Float32}(length(foreground_training_examples))

    poss_bg_train = Vector{Float32}(length(background_training_examples))
    tlens_bg_train   = Vector{Float32}(length(background_training_examples))

    poss_fg_test = Vector{Float32}(length(foreground_testing_examples))
    tlens_fg_test   = Vector{Float32}(length(foreground_testing_examples))

    poss_bg_test = Vector{Float32}(length(background_testing_examples))
    tlens_bg_test   = Vector{Float32}(length(background_testing_examples))

    for (poss, tlens, examples) in
                [(poss_fg_train, tlens_fg_train, foreground_training_examples),
                 (poss_bg_train, tlens_bg_train, background_training_examples),
                 (poss_fg_test, tlens_fg_test, foreground_testing_examples),
                 (poss_bg_test, tlens_bg_test, background_testing_examples) ]
        for (i, example) in enumerate(examples)
            poss[i] = example.fpdist / example.tlen
            tlens[i] = example.tlen
        end
    end

    lenbinsize = 250

    best_loss = Inf
    best_numlenbins = 0
    best_numposbins = 0

    # search for a good number of bins in the histogram
    # for numlenbins in 1:10, numposbins in 1:10
    # for numlenbins in 1:20, numposbins in 1:20
    for numlenbins in 2:10, numposbins in 2:10
        intp = fit_interpolation(
            poss_fg_train, tlens_fg_train,
            poss_bg_train, tlens_bg_train,
            numlenbins, lenbinsize, numposbins)

        model = PositionalBiasModel(intp, numlenbins * lenbinsize)

        loss = compute_loss(
            model,
            poss_fg_test, tlens_fg_test,
            poss_bg_test, tlens_bg_test)

        @show (numlenbins, numposbins, loss)

        if loss < best_loss
            best_loss = loss
            best_numlenbins = numlenbins
            best_numposbins = numposbins
        end
    end

    @show (best_numlenbins, best_numposbins)

    best_numlenbins = 15
    best_numposbins = 15

    intp = fit_interpolation(
        poss_fg_train, tlens_fg_train,
        poss_bg_train, tlens_bg_train,
        best_numlenbins, lenbinsize, best_numposbins,
        normalize_to_mode=true)
    best_model = PositionalBiasModel(intp, best_numlenbins * lenbinsize)

    return best_model::PositionalBiasModel
end


function evaluate(posmodel::PositionalBiasModel, tlen, pos; classification::Bool=false)
    # tlen = min(posmodel.max_len, tlen)
    p = Float32(posmodel.intp[tlen, pos])
    # if tlen > 10000 && p < 0.01
    #     @show (tlen, pos, p)
    # end
    p = clamp(p, 0.01f0, 10.0f0)
    return p
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
        bias_foreground_training_examples,
        bias_background_training_examples,
        bias_foreground_testing_examples,
        bias_background_testing_examples,
        weights)

    bm = BiasModel(seqbias_left, seqbias_right, gc_model, pos_model)


    for (i, example) in enumerate(bias_foreground_training_examples)
        weights[i] = evaluate(bm.pos_model, example.tlen, example.fpdist/example.tlen)
    end

    for (i, example) in enumerate(bias_foreground_training_examples)
        idx = length(bias_foreground_training_examples) + i
        weights[idx] = evaluate(bm.pos_model, example.tlen, example.fpdist/example.tlen)
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
            evaluate(bm.pos_model, example.tlen, example.fpdist/example.tlen)
        bs[idx] = bias
        idx += 1
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.left_seqbias, example.left_seq) *
            evaluate(bm.right_seqbias, example.right_seq) *
            evaluate(bm.gc_model, example.frag_gc) *
            evaluate(bm.pos_model, example.tlen, example.fpdist/example.tlen)
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
            evaluate(bm.pos_model, example.tlen, example.fpdist/example.tlen)
        bs[idx] = bias
        idx += 1
    end

    for example in background_testing_examples
        bias =
            evaluate(bm.pos_model, example.tlen, example.fpdist/example.tlen)
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
        # left_bias[pos] =
        #     evaluate(bm.left_seqbias, tseq, pos)
        # left_bias[pos] =
        #     evaluate(bm.left_seqbias, tseq, pos) *
        #     evaluate(bm.pos_model, tlen, pos/tlen)
        left_bias[pos] =
            evaluate(bm.pos_model, tlen, pos/tlen)
        # left_bias[pos] =
        #     evaluate(bm.left_seqbias, tseq, pos)
        # left_bias[pos] = 1.0
    end

    # if tlen > 10000
    #     println("-----------------------------------------------")
    #     @show tlen
    #     @show left_bias
    # end

    # right bias
    for pos in 1:tlen
        # right_bias[pos] = evaluate(bm.right_seqbias, tseq, pos)
        right_bias[pos] = 1.0
    end
end

