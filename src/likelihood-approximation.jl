
include("randn.jl")

function approximate_likelihood(input_filename::String, output_filename::String)
    sample = read(input_filename, RNASeqSample)
    approximate_likelihood(sample, output_filename)
end


function approximate_likelihood(sample::RNASeqSample, output_filename::String)
    αs, βs, t = approximate_likelihood(sample)
    h5open(output_filename, "w") do out
        n = sample.n
        out["n"] = sample.n
        out["alpha", "compress", 1] = αs[1:n-1]
        out["beta", "compress", 1]  = βs[1:n-1]

        node_parent_idxs = Array{Int32}(length(t.nodes))
        node_js          = Array{Int32}(length(t.nodes))
        for i in 1:length(t.nodes)
            node = t.nodes[i]
            node_parent_idxs[i] = node.parent_idx
            node_js[i] = node.j
        end
        out["node_parent_idxs"] = node_parent_idxs
        out["node_js"] = node_js

        g = g_create(out, "metadata")
        attrs(g)["gfffilename"] = sample.transcript_metadata.filename
        attrs(g)["gffhash"]     = base64encode(sample.transcript_metadata.gffhash)
        attrs(g)["gffsize"]     = sample.transcript_metadata.gffsize
    end
end


function approximate_likelihood_from_rsem(input_filename, output_filename)
    input = open(input_filename)

    mat = match(r"^(\d+)\s+(\d+)", readline(input))
    n = parse(Int, mat.captures[1])
    m0 = parse(Int, mat.captures[2])

    I = Array(Int, 0)
    J = Array(Int, 0)
    V = Array(Float32, 0)
    println("reading rsem likelihood matrix...")
    i = 0
    for line in eachline(input)
        i += 1

        toks = split(line, ' ')
        k = 1
        while k < length(toks)
            j = 1 + parse(Int, toks[k])
            v = parse(Float64, toks[k+1])

            if v > MIN_FRAG_PROB
                push!(I, i)
                push!(J, j)
                push!(V, Float32(v))
            end

            k += 2
        end
    end
    @show maximum(I)
    @show maximum(J)
    m = i
    close(input)
    @show (minimum(I), maximum(I), minimum(J), maximum(J), m, n)
    println("making SparseMatrixCSC")
    X = sparse(I, J, V, m, n)
    println("making SparseMatrixRSB")
    rsbX = SparseMatrixRSB(X)
    println("done")
    @show size(X)

    effective_lengths = ones(Float32, n)
    sample = RNASeqSample(m, n, rsbX, effective_lengths, TranscriptsMetadata())

    @time μ, σ, w = approximate_likelihood(sample)

    h5open(output_filename, "w") do out
        n = sample.n
        out["n"] = sample.n
        out["mu", "compress", 1] = μ[1:n-1]
        out["sigma", "compress", 1] = σ[1:n-1]
        out["w", "compress", 1] = w
    end
end


function approximate_likelihood_from_isolator(input_filename,
                                              effective_lengths_filename,
                                              output_filename)
    input = open(input_filename)

    n = parse(Int, readline(input))

    m = parse(Int, readline(input))
    @show (m, n)
    I = Array{Int}(0)
    J = Array{Int}(0)
    V = Array{Float32}(0)
    println("reading isolator likelihood matrix...")
    for line in eachline(input)
        j_, i_, v_ = split(line, ',')
        i = 1 + parse(Int, i_)
        j = 1 + parse(Int, j_)
        v = parse(Float32, v_)

        push!(I, i)
        push!(J, j)
        push!(V, v)
    end
    X = sparse(I, J, V, m, n)
    rsbX = SparseMatrixRSB(X)
    println("done")

    println("reading isolator transcript weights...")
    effective_lengths = Float32[]
    open(effective_lengths_filename) do input
        for line in eachline(input)
            push!(effective_lengths, parse(Float32, line))
        end
    end
    println("done")

    sample = RNASeqSample(m, n, rsbX, effective_lengths, TranscriptsMetadata())

    # measure and dump connectivity info
    #=
    p = sortperm(I)
    I = I[p]
    J = J[p]
    v = V[p]
    edge_count = Dict{Tuple{Int, Int}, Int}()
    a = 1
    while a <= length(I)
        if I[a] % 1000 == 0
            @show I[a]
        end

        b = a
        while b <= length(I) && I[a] == I[b]
            b += 1
        end

        for k in a:b-1
            for l in k+1:b-1
                key = J[k] < J[l] ? (J[k], J[l]) : (J[l], J[k])
                if haskey(edge_count, key)
                    edge_count[key] += 1
                else
                    edge_count[key] = 1
                end
            end
        end
        a = b
    end

    open("connectivity.csv", "w") do out
        for ((u,v), c) in edge_count
            println(out, u, ",", v, ",", c)
        end
    end
    =#

    #@profile μ, σ = approximate_likelihood(sample)
    #Profile.print()

    @time μ, σ = approximate_likelihood(sample)

    h5open(output_filename, "w") do out
        n = sample.n
        out["n"] = sample.n
        out["mu", "compress", 1] = μ[1:n-1]
        out["sigma", "compress", 1] = σ[1:n-1]
    end
end


"""
Compute the entropy of a multivariate normal distribution along with gradients
wrt σ.

Note: when σ² does not have length divisible by the simd vector length it needs
to be padded with extra 1s
"""
function normal_entropy!(grad, σ, n)
    vs = reinterpret(FloatVec, σ)
    vs_sumlogs = sum(mapreduce(log, +, zero(FloatVec), vs))
    entropy = 0.5 * n * log(2 * π * e) + vs_sumlogs

    return entropy
end


"""
Sample from the variational distribution and evaluate the true likelihood
function at these samples.
"""
function diagnostic_samples(model, X, μ, σ, i)
    n = length(μ)

    num_samples = 500
    samples = Array(Float32, num_samples)
    weights = Array(Float32, num_samples)
    π = Array(Float32, n)
    π_grad = Array(Float32, n)

    for k in 1:num_samples
        for j in 1:n
            π[j] = rand(Normal(μ[j], σ[j]))
        end

        ll = log_likelihood(model, X, π, π_grad)
        samples[k] = model.π_simplex[i]
        weights[k] = ll
    end

    return (samples, weights)
end


function approximate_likelihood{T}(s::RNASeqSample)
    m, n = size(s)

    model = Model(m, n)

    # learning rate
    # adam_learning_rate = 0.02
    adam_learning_rate = 0.1
    adam_learning_rate_decay = 4e-5

    # epsilon
    adam_eps = 1e-8

    # resistance
    # adam_rv = 0.9
    # adam_rm = 0.999
    # adam_rv = 0.8
    # adam_rm = 0.9
    adam_rv = 0.7
    adam_rm = 0.8
    # adam_rv = 0.1
    # adam_rm = 0.1

    # gradient running mean
    m_α = Array{Float32}(n-1)
    m_β = Array{Float32}(n-1)

    # gradient running variances
    v_α = Array{Float32}(n-1)
    v_β = Array{Float32}(n-1)

    # step size clamp
    ss_max_α_step = 1e-1
    ss_max_β_step = 1e-1

    # srand(43241)

    # number of monte carlo samples to estimate gradients an elbo at each
    # iteration
    num_mc_samples = 2

    # cluster transcripts for hierachrical stick breaking
    @time t = HSBTransform(s.X)

    # Unifom distributed values
    zs = Array{Float32}(n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(n)

    # log transformed kumaraswamy parameters
    # αs = zeros(Float32, n-1)
    # βs = zeros(Float32, n-1)
    αs = fill(log(10f0), n-1)
    βs = fill(log(921.7f0), n-1)

    #=
    # choose initial values to avoid underflow
    # count subtree size and store in the node's input_value field
    k = n-1
    nodes = t.nodes
    for i in length(nodes):-1:1
        node = nodes[i]
        if node.j != 0
            node.input_value = 1
        else
            a = beta_a = node.left_child.input_value + 1
            beta_b = node.right_child.input_value + 1

            u = (1 - beta_a) / (2 - beta_a - beta_b)

            @show beta_a
            @show beta_b
            @show u

            b = 1/beta_a + (beta_a - 1) / (beta_a * u^a)
            b = min(b, 1f16)

            αs[k] = log(a)
            βs[k] = log(b)

            @show (exp(αs[k]), exp(βs[k]), node.left_child.input_value,
                   node.right_child.input_value)

            node.input_value =
                1 + node.left_child.input_value + node.right_child.input_value
            k -= 1
        end
    end
    =#

    as = Array{Float32}(n-1) # exp(αs)
    bs = Array{Float32}(n-1) # exp(βs)

    # various intermediate gradients
    α_grad = Array{Float32}(n-1)
    β_grad = Array{Float32}(n-1)
    a_grad = Array{Float32}(n-1)
    b_grad = Array{Float32}(n-1)
    y_grad = Array{Float32}(n-1)
    x_grad = Array{Float32}(n)
    work   = Array{Float32}(n-1) # used by kumaraswamy_transform!

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

    step_num = 0
    small_step_count = 0
    fruitless_step_count = 0

    # stopping criteria
    max_small_steps = 2
    max_fruitless_steps = 20
    max_steps = 200
    minz = eps(Float32)
    maxz = 1.0f0 - eps(Float32)

    println("Optimizing ELBO: ", -Inf)

    # I, J, V = findnz(s.X)
    # X = sparse(I, J, V, m, n)

    tic()

    while true
        tic()
        step_num += 1
        elbo0 = elbo
        elbo = 0.0
        fill!(α_grad, 0.0f0)
        fill!(β_grad, 0.0f0)

        for i in 1:n-1
            as[i] = exp(αs[i])
            bs[i] = exp(βs[i])
        end

        println("-------------------------")

        for _ in 1:num_mc_samples
            fill!(x_grad, 0.0f0)
            fill!(y_grad, 0.0f0)
            fill!(a_grad, 0.0f0)
            fill!(b_grad, 0.0f0)
            for i in 1:n-1
                zs[i] = min(maxz, max(minz, rand()))
            end

            tic()
            kum_ladj = kumaraswamy_transform!(as, bs, zs, ys, work)  # z -> y
            toc() # 0.05 seconds

            tic()
            hsp_ladj = hsb_transform!(t, ys, xs)               # y -> x
            toc() # 0.07 seconds

            tic()
            lp = log_likelihood(model, s.X, s.effective_lengths, xs, x_grad)
            elbo += lp + kum_ladj + hsp_ladj
            toc() # 0.25 seconds

            tic()
            hsb_transform_gradients!(t, ys, y_grad, x_grad)
            toc() # 0.05 seconds

            # negative entropy
            # for i in 1:n-1
            #     y_grad[i] += 1/ys[i]
            # end

            tic()
            kumaraswamy_transform_gradients!(zs, as, bs, y_grad, a_grad, b_grad)
            toc() # 0.06 seconds

            # adjust for log transform and accumulate
            for i in 1:n-1
                α_grad[i] += as[i] * a_grad[i]
                β_grad[i] += bs[i] * b_grad[i]
            end
        end


        for i in 1:n-1
            α_grad[i] /= num_mc_samples
            β_grad[i] /= num_mc_samples
        end

        elbo /= num_mc_samples # get estimated expectation over mc samples

        # @show x_grad
        # @show y_grad
        # @show a_grad
        # @show b_grad
        # @show α_grad
        # @show β_grad

        # Note: the elbo has a negative entropy term is well, but but we are
        # using uniform values on [0,1] which has entropy of 0, so that term
        # goes away.

        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)
        # @printf("\e[F\e[JOptimizing ELBO: %.6e\n", elbo)
        @printf("Optimizing ELBO: %.4e\n", elbo)

        if step_num == 1
                m_α[:] = α_grad
                m_β[:] = β_grad

                v_α[:] = α_grad.^2
                v_β[:] = β_grad.^2
        else
            for i in 1:n-1
                m_α[i] = adam_rm * m_α[i] + (1 - adam_rm) * α_grad[i]
                m_β[i] = adam_rm * m_β[i] + (1 - adam_rm) * β_grad[i]

                v_α[i] = adam_rv * v_α[i] + (1 - adam_rv) * α_grad[i]^2
                v_β[i] = adam_rv * v_β[i] + (1 - adam_rv) * β_grad[i]^2
            end
        end

        @show extrema(v_α)
        @show extrema(m_α)

        @show extrema(v_β)
        @show extrema(m_β)

        max_delta = 0.0
        for i in 1:n-1
            # update a parameters
            m_α_i = m_α[i] / (1 - adam_rm^step_num)
            v_α_i = v_α[i] / (1 - adam_rv^step_num)
            delta = adam_learning_rate * m_α_i / (sqrt(v_α_i) + adam_eps)
            max_delta = max(max_delta, abs(delta))
            αs[i] += clamp(delta, -ss_max_α_step, ss_max_α_step)

            # update b parameters
            m_β_i = m_β[i] / (1 - adam_rm^step_num)
            v_β_i = v_β[i] / (1 - adam_rv^step_num)
            delta = adam_learning_rate * m_β_i / (sqrt(v_β_i) + adam_eps)
            max_delta = max(max_delta, abs(delta))
            βs[i] += clamp(delta, -ss_max_β_step, ss_max_β_step)
        end

        adam_learning_rate *= 1 / (1 + adam_learning_rate_decay * step_num)
        @show adam_learning_rate

        @show max_delta

        toc()

        if elbo < max_elbo
            fruitless_step_count += 1
        else
            fruitless_step_count = 0
        end

        # if fruitless_step_count > 400
        #     break
        # end

        if step_num > 500
        # if step_num > 20
            break
        end
    end

    # Profile.print()
    # exit()

    toc()

    println("Finished in ", step_num, " steps")

    # Write out point estimates for convenience
    #log_likelihood(model, s.X, s.effective_lengths, μ, π_grad)
    #open("point-estimates.csv", "w") do out
        #for i in 1:n
            #@printf(out, "%e\n", model.π_simplex[i])
        #end
    #end

    return αs, βs, t
end


