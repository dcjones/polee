

function expectation_maximization(input_filename, output_filename)
    sample = read(input_filename, RNASeqSample)

    m, n = sample.m, sample.n
    els = sample.effective_lengths

    # gibbs sampler needs these probs weighted by effective length
    Is, Js, Vs = findnz(sample.X)

    # X = convert(SparseMatrixCSC{Float32}, sample.X)
    X = sparse(Is, Js, Vs)
    Xt = SparseMatrixCSC(transpose(X))

    # transcript read count expectations
    cs = Array{Float32}(undef, n)

    # transcript mixture
    ys = zeros(Float32, n)
    ys0 = similar(ys)
    fill!(ys, 1/n)

    # temporary vector to store row products
    wlen = 0
    for i in 1:m
        wlen = max(wlen, Xt.colptr[i+1] - Xt.colptr[i])
    end
    ws = zeros(Float64, wlen)
    fill!(ws, 0.0f0)

    frag_probs = Vector{Float32}(undef, m)
    log_frag_probs = Vector{Float32}(undef, m)

    pAt_mul_B!(frag_probs, Xt, ys)
    log!(log_frag_probs, frag_probs, m)
    lp = sum(log_frag_probs)

    ϵ = 1e-6

    while true
        copyto!(ys0, ys)
        lp0 = lp

        # assign reads
        fill!(cs, 0.0)
        for i in 1:m
            wsum = 0.0
            wcnt = 0
            for (l, k) in enumerate(Xt.colptr[i]:Xt.colptr[i+1]-1)
                j = Xt.rowval[k]
                ws[l] = Float64(Xt.nzval[k]) * Float64(ys[j])
                wsum += Float64(ws[l])
                wcnt += 1
            end

            wlen = Xt.colptr[i+1] - Xt.colptr[i]
            for (l, k) in enumerate(Xt.colptr[i]:Xt.colptr[i+1]-1)
                j = Xt.rowval[k]
                cs[j] += ws[l] / wsum
            end
        end
        cs_sum = sum(cs)

        # set mixture
        ys_sum = 0.0f0
        for j in 1:n
            ys[j] = cs[j] / cs_sum
        end

        pAt_mul_B!(frag_probs, Xt, ys)
        log!(log_frag_probs, frag_probs, m)
        lp = sum(log_frag_probs)
        @show lp

        if lp - lp0 < ϵ
            break
        end
    end

    # effective length transform and convert to tpm
    ys ./= els
    ys ./= sum(ys)
    ys .*= 1e6

    open(output_filename, "w") do output
        println(output, "expression")
        for j in 1:n
            println(output, ys[j])
        end
    end
end