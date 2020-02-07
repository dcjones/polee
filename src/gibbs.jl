
function gibbs_sampler(
        input_filename, output_filename, ts::Transcripts;
        kallisto::Bool=false, num_samples::Int=1000,
        num_burnin_samples = 2000, sample_stride::Int=25,
        convergence_test_stride::Int=125)

    nthreads = Threads.nthreads()

    sample_per_chain = div(num_samples, nthreads)

    sample = read(input_filename, RNASeqSample)
    m, n = sample.m, sample.n

    if n != length(ts)
        error("Likelihood matrix has different number of transcripts than annotations.")
    end

    els = sample.effective_lengths

    # gibbs sampler needs these probs weighted by effective length
    Is, Js, Vs = findnz(sample.X)

    # X = convert(SparseMatrixCSC{Float32}, sample.X)
    X = sparse(Is, Js, Vs)
    Xt = SparseMatrixCSC(transpose(X))

    # read assignments
    zs = Array{Int}(undef, (nthreads, m))

    # transcript read counts
    cs = Array{UInt32}(undef, (nthreads, n))
    # cs = Array{Float64}(nthreads, n)

    # transcript mixture
    ys = Array{Float32}(undef, (nthreads, n))

    # intermediate results for form computing convergence statistics
    chain_means = Vector{Float32}(undef, 2 * nthreads)
    chain_vars = Vector{Float32}(undef, 2 * nthreads)

    # per-transcript convergence stats
    Rs = Vector{Float32}(undef, n)

    # choosing starting positions uniformly at random
    alpha = 1.0
    for t in 1:nthreads
        ys_sum = 0.0f0
        for i in 1:n
            ys[t, i] = rand(Gamma(alpha, 1))
            ys_sum += ys[t, i]
        end
        ys[t,:] ./= ys_sum
    end

    # effective length adjusted mixtures
    xs = similar(ys)

    # temporary vector to store row products
    wlen = 0
    for i in 1:m
        wlen = max(wlen, Xt.colptr[i+1] - Xt.colptr[i])
    end
    ws = Array{Float64}(undef, (nthreads, wlen))
    fill!(ws, 0.0f0)

    samples = Array{Float32}(undef, (nthreads, sample_per_chain, n))

    rngs = [MersenneTwister(Random.make_seed()) for t in 1:nthreads]

    # burn-in
    Threads.@threads for t in 1:nthreads
        for sample_num in 1:num_burnin_samples
            if t == 1 && (sample_num % 10) == 0
                println("Burn-in: ", sample_num, "/", num_burnin_samples)
            end
            generate_gibbs_sample(rngs[t], m, n, t, Xt, els, cs, ws, xs, ys, zs,
                                    samples, 0)
        end
    end

    diagnostics_filename = string(output_filename, ".convergence.csv")
    diagnostics_output = open(diagnostics_filename, "w")

    num_chunks = div(sample_stride * sample_per_chain, convergence_test_stride)
    for l in 1:num_chunks
        Threads.@threads for t in 1:nthreads
            stored_sample_num = div((l-1) * convergence_test_stride, sample_stride)
            for sample_num in 1:convergence_test_stride
                if t == 1 && (sample_num % 10) == 0
                    println("Sampling: ", sample_num + (l-1) * convergence_test_stride,
                            "/", sample_stride * sample_per_chain)
                end

                if (sample_num-1) % sample_stride == 0
                    stored_sample_num += 1
                end

                generate_gibbs_sample(rngs[t], m, n, t, Xt, els, cs, ws, xs, ys, zs,
                                        samples, stored_sample_num)
            end
        end

        convergence_stats(samples, chain_means, chain_vars, Rs, n, nthreads,
                          div(l*convergence_test_stride, sample_stride))
        @show extrema(Rs)
        @show quantile(Rs, [0.0, 1e-3, 1e-2, 0.5, 0.99, 0.999, 1.0])
        println(diagnostics_output, join(Rs, ","))
    end
    close(diagnostics_output)

    function prop_to_counts(prop)
        prop_ = prop .* els
        prop_ ./= sum(prop_)
        return Vector{Float64}(prop_ .* m)
    end

    # print samples
    if kallisto
        h5open(output_filename, "w") do output
            post_mean = mean(samples, dims=(1,2))[1,1,:]
            output["est_counts"] = prop_to_counts(post_mean)

            aux_group = g_create(output, "aux")
            aux_group["num_bootstrap"]    = Int[size(samples, 1) * size(samples, 2)]
            aux_group["eff_lengths"]      = Vector{Float64}(els)
            aux_group["lengths"]          = Int[exonic_length(t) for t in ts]
            aux_group["ids"]              = String[t.metadata.name for t in ts]
            aux_group["call"]             = String[join(ARGS, " ")]
            aux_group["index_version"]    = Int[-1]
            aux_group["kallisto_version"] = "polee debug-sample"
            aux_group["start_time"]       = string(now())

            bootstrap_group = g_create(output, "bootstrap")
            k = 0
            for i in 1:size(samples, 1)
                for j in 1:size(samples, 2)
                    bootstrap_group[string("bs", k)] = prop_to_counts(samples[i, j, :])
                    k += 1
                end
            end
        end
    else
        open(output_filename, "w") do output
            for (i, t) in enumerate(ts)
                print(output, t.metadata.name)
                if i < n
                    print(output, ",")
                end
            end
            println(output)

            for t in 1:size(samples, 1)
                for sample_num in 1:size(samples, 2)
                    for i in 1:size(samples, 3)
                        @printf(output, "%e", samples[t, sample_num, i])
                        if i < n
                            print(output, ",")
                        end
                    end
                    println(output)
                end
            end
        end
    end
end


function generate_gibbs_sample(rng, m, n, t, X, els, cs, ws, xs, ys, zs,
                               samples, stored_sample_num)
    # sample zs
    zs[t,:] .= 0
    for i in 1:m
        wsum = 0.0
        for (l, k) in enumerate(X.colptr[i]:X.colptr[i+1]-1)
            j = X.rowval[k]
            ws[t, l] = Float64(X.nzval[k]) * Float64(ys[t, j])
            wsum += Float64(ws[t, l])
        end

        r = wsum * rand(rng)
        wcsum = 0.0
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
    cs[t, :] .= 0
    for i in 1:m
        if zs[t, i] != 0
            cs[t, zs[t, i]] += 1
        end
    end

    # sample mixture
    ys_sum = 0.0f0
    for j in 1:n
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


function convergence_stats(samples, chain_means, chain_vars, Rs,
                           n, nthreads, sample_count)
    k = div(sample_count, 2)
    mid = div(sample_count + 1, 2)
    splits = (1:mid, mid+1:sample_count)
    @show (sample_count, k, mid, splits)
    for i in 1:n
        chain_num = 1
        for split in splits
            for t in 1:nthreads
                chain_means[chain_num] = mean(samples[t, split, i])
                chain_num += 1
            end
        end

        # total_mean = mean(samples[:, :, i])
        total_mean = mean(chain_means)

        # between sequence variance
        B = (k / (2*nthreads - 1)) * sum((chain_means .- total_mean).^2)

        chain_num = 1
        for split in splits
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
end


function generate_em_iteration(rng, m, n, t, X, els, cs, ws, xs, ys, zs,
                               samples, stored_sample_num)
    # sample zs
    cs[t,:] = 0
    for i in 1:m
        wsum = 0.0f0
        for (l, k) in enumerate(X.colptr[i]:X.colptr[i+1]-1)
            j = X.rowval[k]
            ws[t, l] = X.nzval[k] * ys[t, j]
            wsum += ws[t, l]
        end
        wlen = X.colptr[i+1] - X.colptr[i]


        for l in 1:wlen
            ws[t, l] /= wsum
            if t == 1 && !isfinite(ws[t, l])
                @show (i, ws[t, 1:wlen])
                exit()
            end
        end

        for (l, k) in enumerate(X.colptr[i]:X.colptr[i+1]-1)
            cs[t, X.rowval[k]] += ws[t, l]
            if t == 1 && !isfinite(cs[X.rowval[k]])
                @show (i, ws[t, l])
                exit()
            end
        end
    end

    # if t == 1
    #     @show extrema(cs[t, :])
    #     exit()
    # end

    ys_sum = 0.0f0
    for j in 1:n
        ys[t, j] = cs[t, j]
        ys_sum += ys[t, j]
        if t == 1 && !isfinite(ys_sum)
            @show ys_sum
            @show ys[t, j]
            exit()
        end
    end
    ys[t,:] ./= ys_sum

    if t == 1
        @show ys_sum
        @show extrema(ys[t,:])
    end

    if stored_sample_num > 0
        xs_sum = 0.0
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