

function gibbs_sampler(input_filename, output_filename)
    num_samples = 300
    num_burnin_samples = 300
    sample_stride = 5 # record every nth sample

    sample = read(input_filename, RNASeqSample)

    X = transpose(convert(SparseMatrixCSC{Float32}, sample.X))
    els = sample.effective_lengths
    m, n = sample.m, sample.n

    # read assignments
    zs = Array{Int}(m)

    # transcript read counts
    cs = Array{UInt32}(n)
    
    # transcript mixture
    ys = Array{Float32}(n)
    fill!(ys, 1.0f0/n)

    # effective length adjusted mixtures
    xs = similar(ys)

    # temporary vector to store row products
    wlen = 0
    for i in 1:m
        wlen = max(wlen, X.colptr[i+1] - X.colptr[i])
    end
    ws = Array{Float32}(wlen)

    out = open(output_filename, "w")

    total_sample_num = num_burnin_samples + sample_stride * num_samples
    # @showprogress for sample_num in 1:total_sample_num
    for sample_num in 1:total_sample_num
        # sample zs
        fill!(zs, 0)
        for i in 1:m
            wsum = 0.0f0
            for (l, k) in enumerate(X.colptr[i]:X.colptr[i+1]-1)
                j = X.rowval[k]
                ws[l] = X.nzval[k] * ys[j]
                wsum += ws[l]
            end

            r = wsum * rand()
            wcsum = 0.0f0
            wlen = X.colptr[i+1] - X.colptr[i]
            for (l, k) in enumerate(X.colptr[i]:X.colptr[i+1]-1)
                wcsum += ws[l]
                if r <= wcsum
                    zs[i] = X.rowval[k]
                    break
                end
            end
            @assert zs[i] != 0 || wlen == 0
        end

        # count reads
        fill!(cs, 0)
        for i in 1:m
            if zs[i] != 0
                cs[zs[i]] += 1
            end
        end
        # sample mixture
        for j in 1:n
            ys[j] = rand(Gamma(1.0 + cs[j], 1.0))
        end
        ys ./= sum(ys)

        if sample_num > num_burnin_samples &&
            ((sample_num - num_burnin_samples - 1) % sample_stride) == 0

            for j in 1:n
                xs[j] = ys[j] / els[j]
            end
            xs ./= sum(xs)

            for (i, x) in enumerate(xs)
                @printf(out, "%e", x)
                if i < n
                    print(out, ",")
                end
            end
            print(out, "\n")
        end
    end

    close(out)
end