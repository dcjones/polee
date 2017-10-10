
function gibbs_sampler(input_filename, output_filename)
    nthreads = Threads.nthreads()

    # num_samples = 80
    # sample_per_chain = div(num_samples, nthreads)
    # num_burnin_samples = 100
    # sample_stride = 5 # record every nth sample

    num_samples = 1000
    sample_per_chain = div(num_samples, nthreads)
    # num_burnin_samples = 100
    num_burnin_samples = 2000
    sample_stride = 20 # record every nth sample

    # num_samples = 1000
    # sample_per_chain = div(num_samples, nthreads)
    # num_burnin_samples = 1000
    # sample_stride = 10 # record every nth sample

    sample = read(input_filename, RNASeqSample)

    X = transpose(convert(SparseMatrixCSC{Float32}, sample.X))
    els = sample.effective_lengths
    m, n = sample.m, sample.n

    # read assignments
    zs = Array{Int}(nthreads, m)

    # transcript read counts
    cs = Array{UInt32}(nthreads, n)

    # transcript mixture
    ys = Array{Float32}(nthreads, n)

    # choosing starting positions at mode
    println("Finding approximate mode")
    ys0 = approximate_likelihood(OptimizeHSBApprox(), transpose(X), Val{true}, num_steps=100)["x"]
    ys0_el = ys0 .* els
    ys0_el ./= sum(ys0_el)
    @show ys0[39073]
    # @show ys0_el[39073]
    # @show els[39073]
    # exit()
    for t in 1:nthreads
        ys[t, :] = ys0
    end

    # choosing starting positions uniformly at random
    # alpha = 1.0
    # for t in 1:nthreads
    #     ys_sum = 0.0f0
    #     for i in 1:n
    #         ys[t, i] = rand(Gamma(alpha, 1))
    #         ys_sum += ys[t, i]
    #     end
    #     ys[t,:] ./= ys_sum
    # end

    # effective length adjusted mixtures
    xs = similar(ys)

    # temporary vector to store row products
    wlen = 0
    for i in 1:m
        wlen = max(wlen, X.colptr[i+1] - X.colptr[i])
    end
    ws = Array{Float32}(nthreads, wlen)

    samples = Array{Float32}(nthreads, sample_per_chain, n)
    # stored_sample_num = 0

    rngs = [srand() for t in 1:nthreads]

    total_sample_num = num_burnin_samples + sample_stride * sample_per_chain

    # @showprogress for sample_num in 1:total_sample_num
    #     if sample_num > num_burnin_samples &&
    #         ((sample_num - num_burnin_samples - 1) % sample_stride) == 0
    #         stored_sample_num += 1
    #     end

    #     # Threads.@threads for t in 1:nthreads
    #     for t in 1:nthreads
    #         generate_gibbs_sample(rngs[t], m, n, t, X, els, cs, ws, xs, ys, zs,
    #                               samples, stored_sample_num)
    #     end
    # end

    Threads.@threads for t in 1:nthreads
    # for t in 1:nthreads
        stored_sample_num = 0
        for sample_num in 1:total_sample_num
            if t == 1 && (sample_num % 10) == 0
                @show ys[t, 39073]
                println(sample_num, "/", total_sample_num)
            end

            if sample_num > num_burnin_samples &&
                ((sample_num - num_burnin_samples - 1) % sample_stride) == 0
                stored_sample_num += 1
            end

            generate_gibbs_sample(rngs[t], m, n, t, X, els, cs, ws, xs, ys, zs,
                                  samples, stored_sample_num)
        end
    end

    @show all(isfinite.(samples))

    # compute convergence statistics
    chain_means = Vector{Float32}(2 * nthreads)
    chain_vars = Vector{Float32}(2 * nthreads)
    k = div(sample_per_chain, 2)

    # convergence stats
    Rs = Vector{Float32}(n)
    for i in 1:n
        chain_num = 1
        for split in (1:k, k+1:sample_per_chain)
            for t in 1:nthreads
                chain_means[chain_num] = mean(samples[t, split, i])
                chain_num += 1
            end
        end
        total_mean = mean(samples[:, :, i])

        # between sequence variance
        B = (k / (2*nthreads - 1)) * sum((chain_means .- total_mean).^2)

        chain_num = 1
        for split in (1:k, k+1:sample_per_chain)
            for t in 1:nthreads
                chain_vars[chain_num] = (1/k) * sum((samples[t, split, i] .- chain_means[chain_num]).^2)
                chain_num += 1
            end
        end

        # within sequence variance
        W = (1 / (2*nthreads)) * sum(chain_vars)

        var = ((k-1)/k) * W + (1/k) * B

        Rs[i] = sqrt(var / W)
    end
    @show extrema(Rs)
    @show quantile(sort(Rs), [0.01, 0.1, 0.5,  0.9, 0.99])

    # print samples
    out = open(output_filename, "w")
    for t in 1:size(samples, 1)
        for sample_num in 1:size(samples, 2)
            for i in 1:size(samples, 3)
                @printf(out, "%e", samples[t, sample_num, i])
                if i < n
                    print(out, ",")
                end
            end
            print(out, "\n")
        end
    end
    close(out)
end


function generate_gibbs_sample(rng, m, n, t, X, els, cs, ws, xs, ys, zs,
                               samples, stored_sample_num)
    # sample zs
    zs[t,:] = 0
    for i in 1:m
        wsum = 0.0f0
        for (l, k) in enumerate(X.colptr[i]:X.colptr[i+1]-1)
            j = X.rowval[k]
            ws[t, l] = X.nzval[k] * ys[t, j]
            wsum += ws[t, l]
        end

        r = wsum * rand(rng)
        wcsum = 0.0f0
        wlen = X.colptr[i+1] - X.colptr[i]
        for (l, k) in enumerate(X.colptr[i]:X.colptr[i+1]-1)
            wcsum += ws[t, l]
            if r <= wcsum
                zs[t, i] = X.rowval[k]
                break
            end
        end
        @assert zs[t, i] != 0 || wlen == 0
    end

    # count reads
    cs[t, :] = 0
    for i in 1:m
        if zs[t, i] != 0
            cs[t, zs[t, i]] += 1
        end
    end

    # sample mixture
    ys_sum = 0.0f0
    for j in 1:n
        # ys[t, j] = rand(Gamma(1.0 + cs[t, j], 1.0))
        ys[t, j] = rand_gamma(rng, 1.0 + cs[t, j], 1.0)
        ys_sum += ys[t, j]
    end
    ys[t,:] ./= ys_sum

    if stored_sample_num > 0
        xs_sum = 0.0f0
        for j in 1:n
            xs[t, j] = ys[t, j] / els[j]
            xs_sum += xs[t, j]
        end
        xs[t,:] ./= xs_sum

        for i in 1:n
            samples[t, stored_sample_num, i] = xs[t, i]
        end

        if t == 1
            @show (samples[t, stored_sample_num, 39073], ys[t, 39073])
        end
    end
end


# This function is adapted from a version by John D. Cook, taken from:
# https://www.johndcook.com/julia_rng.html
function rand_gamma(rng, shape, scale)
    if shape <= 0.0
        error("Shape parameter must be positive")
    end
    if scale <= 0.0
        error("Scale parameter must be positive")
    end

    ## Implementation based on "A Simple Method for Generating Gamma Variables"
    ## by George Marsaglia and Wai Wan Tsang.
    ## ACM Transactions on Mathematical Software
    ## Vol 26, No 3, September 2000, pages 363-372.

    if shape >= 1.0
        d = shape - 1.0/3.0
        c = 1.0/sqrt(9.0*d)
        while true
            x = randn(rng)
            v = 1.0 + c*x
            while v <= 0.0
                x = randn(rng)
                v = 1.0 + c*x
            end
            v = v*v*v
            u = rand(rng)
            xsq = x*x
            if u < 1.0 -.0331*xsq*xsq || log(u) < 0.5*xsq + d*(1.0 - v + log(v))
                return scale*d*v
            end
        end
    else
        g = rand_gamma(rng, shape+1.0, 1.0)
        w = rand(rng)
        return scale*g*pow(w, 1.0/shape)
    end
end
