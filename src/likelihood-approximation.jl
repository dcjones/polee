

abstract type LikelihoodApproximation end

"""
Optimize point estimates using HSB transformation.
"""
struct OptimizeHSBApprox <: LikelihoodApproximation end

"""
Logistic-normal distribution.
"""
struct LogisticNormalApprox <: LikelihoodApproximation end

"""
Logit hierarchical stick breaking with logit Skew-Normal distribution.
"""
struct LogitSkewNormalHSBApprox <: LikelihoodApproximation
    treemethod::Symbol # [one of :sequential, :random, :clustered]
end
LogitSkewNormalHSBApprox() = LogitSkewNormalHSBApprox(:clustered)

"""
Hierarchical stick breaking with Kumaraswamy distributed balances.
"""
struct KumaraswamyHSBApprox <: LikelihoodApproximation
    treemethod::Symbol # [one of :sequential, :random, :clustered]
end
KumaraswamyHSBApprox() = KumaraswamyHSBApprox(:clustered)

"""
Hierarchical stick breaking with Logit-Normal distributed balances.
"""
struct LogitNormalHSBApprox <: LikelihoodApproximation
    treemethod::Symbol # [one of :sequential, :random, :clustered]
end
LogitNormalHSBApprox() = LogitNormalHSBApprox(:clustered)

"""
Isometric Log-ratio transformed normal distribution.
"""
struct NormalILRApprox <: LikelihoodApproximation
    treemethod::Symbol # [one of :sequential, :random, :clustered]
end
NormalILRApprox() = NormalILRApprox(:clustered)

"""
Additive Log-ratio transformed normal distribution.
"""
struct NormalALRApprox <: LikelihoodApproximation
    """
    What element should be diviser. If 0, a random index will be chosen, if -1,
    the last index will be chosen.
    """
    reference_idx::Int
end
NormalALRApprox() = NormalALRApprox(-1)


"""
Structure used for serializing likelihood approximation data and metadata.
"""
mutable struct ApproximatedLikelihoodData
    mu::Vector{Float32}
    omega::Vector{Float32}
    alpha::Vector{Float32}
    efflens::Vector{Float32}

    parent_idxs::Vector{Int32}
    leaf_idxs::Vector{Int32}

    approx_type::String
    gff_hash::Vector{UInt8}
    gff_filename::String
    gff_size::Int32
end


function Base.write(output::IO, data::ApproximatedLikelihoodData, format_version)
    zlib_output = ZlibDeflateOutputStream(output)
    write(zlib_output, Int32(format_version))
    fb = FlatBuffers.build!(data)
    write(zlib_output, view(fb.bytes, (fb.head + 1):length(fb.bytes)))
end


function read_approximated_likelihood_data(input_filename)
    zlib_input = ZlibInflateInputStream(open(input_filename))
    format_version = read(zlib_input, Int32)
    bytes = read(zlib_input)
    root_pos = read(IOBuffer(bytes[1:4]), Int32)
    return FlatBuffers.read(ApproximatedLikelihoodData, bytes, root_pos)
end


# Other approximations we could implement, but would require a different
# optimization method.
#   - Dirichlet
#   - HSB with Beta distributed balances

function approximate_likelihood(approximation::LikelihoodApproximation,
                                input_filename::String, output_filename::String,
                                output_format::String)
    sample = read(input_filename, RNASeqSample)
    approximate_likelihood(approximation, sample, output_filename)
end


function approximate_likelihood(approximation::LikelihoodApproximation,
                                sample::RNASeqSample, output_filename::String,
                                output_format::String)
    params = approximate_likelihood(approximation, sample.X)

    if output_format == "hdf5"
        h5open(output_filename, "w") do out
            n = sample.n
            out["n"] = sample.n
            out["effective_lengths", "compress", 1] = sample.effective_lengths

            for (key, val) in params
                out[key, "compress", 1] = val
            end

            g = g_create(out, "metadata")
            attrs(g)["approximation"] = string(typeof(approximation))
            attrs(g)["gfffilename"] = sample.transcript_metadata.filename
            attrs(g)["gffhash"]     = base64encode(sample.transcript_metadata.gffhash)
            attrs(g)["gffsize"]     = sample.transcript_metadata.gffsize
        end
    elseif output_format == "flatbuffer"
        data = ApproximatedLikelihoodData(
            params["mu"],
            params["omega"],
            params["alpha"],
            sample.effective_lengths,
            params["node_parent_idxs"],
            params["node_js"],
            string(typeof(approximation)),
            sample.transcript_metadata.gffhash,
            sample.transcript_metadata.filename,
            sample.transcript_metadata.gffsize)

        @show sum(data.mu .== 0.0)
        open(output_filename, "w") do output
            write(output, data, PREPARED_SAMPLE_FORMAT_VERSION)
        end
    else
        error("$(output_format) is not a supported output format")
    end
end


function adam_learning_rate(step_num)
    return ADAM_INITIAL_LEARNING_RATE * exp(-ADAM_LEARNING_RATE_DECAY * step_num)
end


function adam_update_mv!(ms, vs, grad, step_num)
    @assert length(ms) == length(vs) == length(grad)
    n = length(ms)
    if step_num == 1
        for i in 1:n
            ms[i] = grad[i]
            vs[i] = grad[i]^2
        end
    else
        for i in 1:n
            ms[i] = ADAM_RM * ms[i] + (1 - ADAM_RM) * grad[i]
            vs[i] = ADAM_RV * vs[i] + (1 - ADAM_RV) * grad[i]^2
        end
    end
end


function adam_update_params!(params, ms, vs, learning_rate, step_num, max_step_size)
    m_denom = (1 - ADAM_RM^step_num)
    v_denom = (1 - ADAM_RV^step_num)

    for i in 1:length(params)
        param_m = ms[i] / m_denom
        param_v = vs[i] / v_denom
        delta = learning_rate * param_m / (sqrt(param_v) + ADAM_EPS)
        params[i] += clamp(delta, -max_step_size, max_step_size)
    end
end


# Some debugging utilities
function node_label(node)
    if node.j != 0
        return @sprintf("L%d", node.j)
    else
        return @sprintf("I%d", node.k)
    end
end


function show_node(node, xs, ys)
    if node.j != 0
        println(node_label(node), "[label=\"x=", xs[node.j], "\"];")
    else
        println(node_label(node), "[label=\"", node.k, "\\ny=", round(ys[node.k], 4), "\\nin=", node.input_value, "\"];")
        println(node_label(node), " -> ", node_label(node.left_child), ";")
        println(node_label(node), " -> ", node_label(node.right_child), ";")
        show_node(node.left_child, xs, ys)
        show_node(node.right_child, xs, ys)
    end
end



function approximate_likelihood{GRADONLY}(::OptimizeHSBApprox, X::SparseMatrixCSC,
                                          ::Type{Val{GRADONLY}}=Val{true})
    m, n = size(X)
    Xt = transpose(X)

    model = Model(m, n)

    # cluster transcripts for hierachrical stick breaking
    t = HSBTransform(X, :sequential)

    m_z = Array{Float32}(n-1)
    v_z = Array{Float32}(n-1)

    ss_max_z_step = 1e-1

    zs = Array{Float32}(n-1)

    # logistic transformed zs values
    ys = Array{Float64}(n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(n)

    z_grad = Array{Float64}(n-1)
    y_grad = Array{Float64}(n-1)
    x_grad = Array{Float64}(n)

    # initial values for y
    k = 1
    nodes = t.nodes
    for i in 1:length(nodes)
        node = nodes[i]
        if node.j != 0
            node.input_value = 1
        else
            nl = node.left_child.subtree_size
            nr = node.right_child.subtree_size
            zs[k] = logit(nl / (nl + nr))
            k += 1
        end
    end

    eps = 1e-10

    prog = Progress(LIKAP_NUM_STEPS, 0.25, "Optimizing ", 60)
    for step_num in 1:LIKAP_NUM_STEPS
        learning_rate = adam_learning_rate(step_num - 1)

        for i in 1:n-1
            ys[i] = logistic(zs[i])
        end

        fill!(x_grad, 0.0f0)
        fill!(y_grad, 0.0f0)

        hsb_ladj = hsb_transform!(t, ys, xs, Val{GRADONLY})
        xs = clamp!(xs, eps, 1 - eps)

        log_likelihood(model.frag_probs, model.log_frag_probs,
                       X, Xt, xs, x_grad, Val{GRADONLY})

        hsb_transform_gradients!(t, ys, y_grad, x_grad)

        for i in 1:n-1
            dy_dz = ys[i] * (1 - ys[i])
            z_grad[i] = dy_dz * y_grad[i]

            # log jacobian gradient
            expz = exp(zs[i])
            z_grad[i] += (1 - expz) / (1 + expz)
        end

        adam_update_mv!(m_z, v_z, z_grad, step_num)
        adam_update_params!(zs, m_z, v_z, learning_rate, step_num, ss_max_z_step)

        # @show xs[30896]
        # @show xs[1]
        # @show xs[end]

        next!(prog)
    end

    for i in 1:n-1
        ys[i] = logistic(zs[i])
    end
    hsb_ladj = hsb_transform!(t, ys, xs, Val{GRADONLY})
    xs = clamp!(xs, eps, 1 - eps)
    return Dict("x" => xs)
end


function approximate_likelihood{GRADONLY}(::LogisticNormalApprox, X::SparseMatrixCSC,
                                          ::Type{Val{GRADONLY}}=Val{false})
    m, n = size(X)
    Xt = transpose(X)
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(n-1)
    m_omega = Array{Float32}(n-1)

    # gradient running variances
    v_mu    = Array{Float32}(n-1)
    v_omega = Array{Float32}(n-1)

    # step size clamp
    ss_max_mu_step    = 2e-2
    ss_max_omega_step = 2e-2

    # Normal distributed values
    zs = Array{Float32}(n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(n)

    mu = fill(0.0f0, n-1)
    omega = fill(0.1f0, n-1)

    # exp(omega)
    sigma = Array{Float32}(n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(n-1)
    omega_grad = Array{Float32}(n-1)
    sigma_grad = Array{Float32}(n-1)
    y_grad = Array{Float32}(n-1)
    x_grad = Array{Float64}(n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

    tic()
    prog = Progress(LIKAP_NUM_STEPS, 0.25, "Optimizing ", 60)
    for step_num in 1:LIKAP_NUM_STEPS
        learning_rate = adam_learning_rate(step_num - 1)

        elbo0 = elbo
        elbo = 0.0
        fill!(mu_grad, 0.0f0)
        fill!(omega_grad, 0.0f0)

        for i in 1:n-1
            sigma[i] = exp(omega[i])
        end

        eps = 1e-10

        for _ in 1:LIKAP_NUM_MC_SAMPLES
            fill!(x_grad, 0.0f0)
            fill!(y_grad, 0.0f0)
            fill!(sigma_grad, 0.0f0)

            # generate log-normal
            zsum = 1.0f0
            for i in 1:n-1
                zs[i] = randn(Float32)
                ys[i] = zs[i] * sigma[i] + mu[i]
            end

            # transform to multivariate logit-normal
            xs[n] = 1.0f0
            exp_y_sum = 1.0f0
            for i in 1:n-1
                xs[i] = exp(ys[i])
                exp_y_sum += xs[i]
            end
            for i in 1:n
                xs[i] /= exp_y_sum
            end
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val{GRADONLY})

            c = 0.0f0
            for i in 1:n
                c += xs[i] * x_grad[i]
            end

            for i in 1:n-1
                y_grad[i] = xs[i] * x_grad[i] - xs[i] * c
            end

            # ladj gradients
            xsum = 0.0f0
            for i in 1:n-1
                xsum += xs[i]
            end

            for i in 1:n-1
                y_grad[i] += (1 / (1 + xsum)) * (xs[i] - xs[i] * xsum) + 1 - xs[i] * (n-1)
            end

            ln_ladj = log(1 + xsum)
            for i in 1:n-1
                ln_ladj += log(xs[i])
            end

            elbo = lp + ln_ladj

            for i in 1:n-1
                mu_grad[i] += y_grad[i]
                sigma_grad[i] = zs[i] * y_grad[i]
            end

            # adjust for log transform and accumulate
            for i in 1:n-1
                omega_grad[i] += sigma[i] * sigma_grad[i]
            end
        end

        for i in 1:n-1
            mu_grad[i]    /= LIKAP_NUM_MC_SAMPLES
            omega_grad[i] /= LIKAP_NUM_MC_SAMPLES
        end

        elbo /= LIKAP_NUM_MC_SAMPLES # get estimated expectation over mc samples

        # Note: the elbo has a negative entropy term is well, but but we are
        # using uniform values on [0,1] which has entropy of 0, so that term
        # goes away.

        @show elbo

        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)

        next!(prog)
    end

    toc()

    return Dict(
        "mu" => mu,
        "omega" => omega)
end




function approximate_likelihood{GRADONLY}(approx::LogitNormalHSBApprox,
                                          X::SparseMatrixCSC,
                                          ::Type{Val{GRADONLY}}=Val{true})
    m, n = size(X)
    Xt = transpose(X)
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(n-1)
    m_omega = Array{Float32}(n-1)

    # gradient running variances
    v_mu    = Array{Float32}(n-1)
    v_omega = Array{Float32}(n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1

    # cluster transcripts for hierachrical stick breaking
    t = HSBTransform(X, approx.treemethod)

    # Unifom distributed values
    zs = Array{Float32}(n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(n)

    hsb_inverse_transform!(t, t.x0, ys)
    mu    = fill(0.0f0, n-1)
    k = 1
    for node in t.nodes
        if node.j == 0
            # nl = node.left_child.subtree_size
            # nr = node.right_child.subtree_size
            # mu[k] = logit(nl / (nl + nr))
            mu[k] = logit(ys[k])
            k += 1
        end
    end

    omega = fill(log(0.1f0), n-1)

    # exp(omega)
    sigma = Array{Float32}(n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(n-1)
    omega_grad = Array{Float32}(n-1)
    sigma_grad = Array{Float32}(n-1)
    y_grad = Array{Float32}(n-1)
    x_grad = Array{Float64}(n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

    tic()
    prog = Progress(LIKAP_NUM_STEPS, 0.25, "Optimizing ", 60)
    for step_num in 1:LIKAP_NUM_STEPS
        learning_rate = adam_learning_rate(step_num - 1)

        elbo0 = elbo
        elbo = 0.0
        fill!(mu_grad, 0.0f0)
        fill!(omega_grad, 0.0f0)

        for i in 1:n-1
            sigma[i] = exp(omega[i])
        end

        eps = 1e-10

        for _ in 1:LIKAP_NUM_MC_SAMPLES
            fill!(x_grad, 0.0f0)
            fill!(y_grad, 0.0f0)
            fill!(sigma_grad, 0.0f0)

            for i in 1:n-1
                zs[i] = randn(Float32)
            end

            ln_ladj = logit_normal_transform!(mu, sigma, zs, ys, Val{GRADONLY})
            ys = clamp!(ys, eps, 1 - eps)

            hsb_ladj = hsb_transform!(t, ys, xs, Val{GRADONLY})                     # y -> x
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val{GRADONLY})
            elbo = lp + ln_ladj + hsb_ladj

            hsb_transform_gradients!(t, ys, y_grad, x_grad)
            logit_normal_transform_gradients!(zs, ys, mu, sigma, y_grad, mu_grad, sigma_grad)

            # adjust for log transform and accumulate
            for i in 1:n-1
                omega_grad[i] += sigma[i] * sigma_grad[i]
            end
        end

        for i in 1:n-1
            mu_grad[i]    /= LIKAP_NUM_MC_SAMPLES
            omega_grad[i] /= LIKAP_NUM_MC_SAMPLES
        end

        elbo /= LIKAP_NUM_MC_SAMPLES # get estimated expectation over mc samples

        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)

        next!(prog)
    end

    toc()

    return merge(flattened_tree(t),
                 Dict{String, Vector}("mu" => mu, "omega" => omega))
end


function approximate_likelihood{GRADONLY}(approx::LogitSkewNormalHSBApprox,
                                          X::SparseMatrixCSC,
                                          ::Type{Val{GRADONLY}}=Val{true})
    m, n = size(X)
    Xt = transpose(X)
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(n-1)
    m_omega = Array{Float32}(n-1)
    m_alpha = Array{Float32}(n-1)

    # gradient running variances
    v_mu    = Array{Float32}(n-1)
    v_omega = Array{Float32}(n-1)
    v_alpha = Array{Float32}(n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1
    ss_max_alpha_step = 2e-2

    # cluster transcripts for hierachrical stick breaking
    t = HSBTransform(X, approx.treemethod)

    # Unifom distributed values
    zs0 = Array{Float32}(n-1)
    zs  = Array{Float32}(n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(n)

    hsb_inverse_transform!(t, t.x0, ys)
    mu    = fill(0.0f0, n-1)
    k = 1
    for node in t.nodes
        if node.j == 0
            mu[k] = logit(ys[k])
            k += 1
        end
    end

    omega = fill(log(0.1f0), n-1)
    alpha = fill(0.0f0, n-1)

    # exp(omega)
    sigma = Array{Float32}(n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(n-1)
    omega_grad = Array{Float32}(n-1)
    sigma_grad = Array{Float32}(n-1)
    alpha_grad = Array{Float32}(n-1)
    y_grad = Array{Float32}(n-1)
    x_grad = Array{Float64}(n)
    z_grad = Array{Float32}(n)

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

    tic()
    prog = Progress(LIKAP_NUM_STEPS, 0.25, "Optimizing ", 60)
    for step_num in 1:LIKAP_NUM_STEPS
        learning_rate = adam_learning_rate(step_num - 1)

        elbo0 = elbo
        elbo = 0.0
        fill!(mu_grad, 0.0f0)
        fill!(omega_grad, 0.0f0)
        fill!(alpha_grad, 0.0f0)

        for i in 1:n-1
            sigma[i] = exp(omega[i])
        end

        eps = 1e-10

        for _ in 1:LIKAP_NUM_MC_SAMPLES
            fill!(x_grad, 0.0f0)
            fill!(y_grad, 0.0f0)
            fill!(z_grad, 0.0f0)
            fill!(sigma_grad, 0.0f0)

            for i in 1:n-1
                zs0[i] = randn(Float32)
            end

            skew_ladj = sinh_asinh_transform!(alpha, zs0, zs, Val{GRADONLY}) # 0.015 seconds
            ln_ladj = logit_normal_transform!(mu, sigma, zs, ys, Val{GRADONLY}) # 0.004 seconds
            ys = clamp!(ys, eps, 1 - eps)

            hsb_ladj = hsb_transform!(t, ys, xs, Val{GRADONLY}) # 0.023 seconds
            xs = clamp!(xs, eps, 1 - eps)

            # @show mu[1:10]
            # @show sigma[1:10]
            # @show alpha[1:10]
            # @show sum(xs)
            # @show xs[[133568, 133569, 133570]]
            # @show xs[1:10]
            # @show xs[133560:133570]


            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val{GRADONLY}) # 0.047 seconds

            elbo = lp + skew_ladj + ln_ladj + hsb_ladj

            hsb_transform_gradients!(t, ys, y_grad, x_grad) # 0.025 seconds
            logit_normal_transform_gradients!(zs, ys, mu, sigma, y_grad, z_grad, mu_grad, sigma_grad) # 0.003 seconds
            sinh_asinh_transform!(zs0, zs, alpha, z_grad, alpha_grad) # 0.027 seconds

            # adjust for log transform and accumulate
            for i in 1:n-1
                omega_grad[i] += sigma[i] * sigma_grad[i]
            end
        end

        for i in 1:n-1
            mu_grad[i]    /= LIKAP_NUM_MC_SAMPLES
            omega_grad[i] /= LIKAP_NUM_MC_SAMPLES
            alpha_grad[i] /= LIKAP_NUM_MC_SAMPLES
        end

        elbo /= LIKAP_NUM_MC_SAMPLES # get estimated expectation over mc samples

        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)
        adam_update_mv!(m_alpha, v_alpha, alpha_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)
        adam_update_params!(alpha, m_alpha, v_alpha, learning_rate, step_num, ss_max_alpha_step)

        next!(prog)
    end

    toc()

    return merge(flattened_tree(t),
                 Dict{String, Vector}("mu" => mu, "omega" => omega, "alpha" => alpha))
end


function approximate_likelihood{GRADONLY}(approx::KumaraswamyHSBApprox,
                                          X::SparseMatrixCSC,
                                          ::Type{Val{GRADONLY}}=Val{true})
    m, n = size(X)
    Xt = transpose(X)
    model = Model(m, n)

    # gradient running mean
    m_α = Array{Float32}(n-1)
    m_β = Array{Float32}(n-1)

    # gradient running variances
    v_α = Array{Float32}(n-1)
    v_β = Array{Float32}(n-1)

    # step size clamp
    ss_max_α_step = 1e-1
    ss_max_β_step = 1e-1

    # cluster transcripts for hierachrical stick breaking
    t = HSBTransform(X, approx.treemethod)

    # Unifom distributed values
    zs = Array{Float32}(n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(n)

    # log transformed kumaraswamy parameters
    αs = zeros(Float32, n-1)
    βs = zeros(Float32, n-1)

    as = Array{Float32}(n-1) # exp(αs)
    bs = Array{Float32}(n-1) # exp(βs)

    # various intermediate gradients
    α_grad = Array{Float32}(n-1)
    β_grad = Array{Float32}(n-1)
    a_grad = Array{Float32}(n-1)
    b_grad = Array{Float32}(n-1)
    y_grad = Array{Float32}(n-1)
    x_grad = Array{Float64}(n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

    # mark the step in which we first find a solution with finite gradients
    first_finite_step = 0

    # stopping criteria
    minz = eps(Float32)
    maxz = 1.0f0 - eps(Float32)

    # println("Optimizing ELBO: ", -Inf)

    # choose initial values to avoid underflow
    # count subtree size and store in the node's input_value field
    tic()
    k = 1
    nodes = t.nodes
    for i in 1:length(nodes)
        node = nodes[i]
        if node.j != 0
            node.input_value = 1
        else
            nl = node.left_child.subtree_size
            nr = node.right_child.subtree_size

            mean = max(min(nl / (nl + nr), 0.99), 0.01)
            var = 0.00001

            αs[k], βs[k] = kumaraswamy_fit_median_var(mean, var)

            k += 1
        end
    end
    toc()

    tic()

    prog = Progress(LIKAP_NUM_STEPS, 0.25, "Optimizing ", 60)
    for step_num in 1:LIKAP_NUM_STEPS
        learning_rate = adam_learning_rate(step_num - 1)

        elbo0 = elbo
        elbo = 0.0
        fill!(α_grad, 0.0f0)
        fill!(β_grad, 0.0f0)

        for i in 1:n-1
            as[i] = exp(αs[i])
            bs[i] = exp(βs[i])
        end

        for _ in 1:LIKAP_NUM_MC_SAMPLES
            fill!(x_grad, 0.0f0)
            fill!(y_grad, 0.0f0)
            fill!(a_grad, 0.0f0)
            fill!(b_grad, 0.0f0)
            for i in 1:n-1
                zs[i] = min(maxz, max(minz, rand()))
            end

            kum_ladj = kumaraswamy_transform!(as, bs, zs, ys, work, Val{GRADONLY})  # z -> y
            ys = clamp!(ys, LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)

            hsb_ladj = hsb_transform!(t, ys, xs, Val{GRADONLY})                     # y -> x
            xs = clamp!(xs, LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val{GRADONLY})
            elbo = lp + kum_ladj + hsb_ladj

            hsb_transform_gradients!(t, ys, y_grad, x_grad)
            kumaraswamy_transform_gradients!(zs, as, bs, y_grad, a_grad, b_grad)

            # adjust for log transform and accumulate
            for i in 1:n-1
                α_grad[i] += as[i] * a_grad[i]
                β_grad[i] += bs[i] * b_grad[i]
            end
        end

        for i in 1:n-1
            α_grad[i] /= LIKAP_NUM_MC_SAMPLES
            β_grad[i] /= LIKAP_NUM_MC_SAMPLES
        end

        elbo /= LIKAP_NUM_MC_SAMPLES # get estimated expectation over mc samples

        # Note: the elbo has a negative entropy term is well, but but we are
        # using uniform values on [0,1] which has entropy of 0, so that term
        # goes away.

        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)

        adam_update_mv!(m_α, v_α, α_grad, step_num)
        adam_update_mv!(m_β, v_β, β_grad, step_num)

        adam_update_params!(αs, m_α, v_α, learning_rate, step_num, ss_max_α_step)
        adam_update_params!(βs, m_β, v_β, learning_rate, step_num, ss_max_β_step)

        next!(prog)
    end

    toc()

    return merge(flattened_tree(t),
                 Dict{String, Vector}("alpha" => αs, "beta" => βs))
end



function approximate_likelihood{GRADONLY}(approx::NormalILRApprox,
                                          X::SparseMatrixCSC,
                                          ::Type{Val{GRADONLY}}=Val{false})
    m, n = size(X)
    Xt = transpose(X)
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(n-1)
    m_omega = Array{Float32}(n-1)

    # gradient running variances
    v_mu    = Array{Float32}(n-1)
    v_omega = Array{Float32}(n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1

    # cluster transcripts for hierachrical stick breaking
    t = ILRTransform(X, approx.treemethod)

    # standard normal values
    zs = Array{Float32}(n-1)

    # destandardized normal random numbers
    ys = Array{Float64}(n-1)

    # ys transformed to simplex using ILR
    xs = Array{Float32}(n)

    mu = fill(0.0f0, n-1)
    omega = fill(log(0.1f0), n-1)

    sigma = Array{Float32}(n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(n-1)
    omega_grad = Array{Float32}(n-1)
    sigma_grad = Array{Float32}(n-1)
    y_grad = Array{Float32}(n-1)
    x_grad = Array{Float32}(n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0

    tic()
    prog = Progress(LIKAP_NUM_STEPS, 0.25, "Optimizing ", 60)
    for step_num in 1:LIKAP_NUM_STEPS
        learning_rate = adam_learning_rate(step_num - 1)

        elbo = 0.0
        fill!(mu_grad, 0.0f0)
        fill!(omega_grad, 0.0f0)

        for i in 1:n-1
            sigma[i] = exp(omega[i])
        end

        eps = 1e-10

        for _ in 1:LIKAP_NUM_MC_SAMPLES
            fill!(x_grad, 0.0f0)
            fill!(y_grad, 0.0f0)
            fill!(sigma_grad, 0.0f0)

            for i in 1:n-1
                zs[i] = randn(Float32)
                ys[i] = mu[i] + sigma[i] * zs[i]
            end

            ilr_ladj = ilr_transform!(t, ys, xs, Val{GRADONLY})                     # y -> x
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val{GRADONLY})
            elbo += lp + ilr_ladj

            ilr_transform_gradients!(t, xs, y_grad, x_grad)

            for i in 1:n-1
                mu_grad[i]    += y_grad[i]
                sigma_grad[i] = zs[i] * y_grad[i]
            end

            # adjust for log transform and accumulate
            for i in 1:n-1
                omega_grad[i] += sigma[i] * sigma_grad[i]
            end
        end

        for i in 1:n-1
            mu_grad[i]    /= LIKAP_NUM_MC_SAMPLES
            omega_grad[i] /= LIKAP_NUM_MC_SAMPLES
            # omega_grad[i] += 1 # entropy gradient
        end

        elbo /= LIKAP_NUM_MC_SAMPLES # get estimated expectation over mc samples
        # elbo += normal_entropy(sigma)

        @show elbo
        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)

        next!(prog)
    end

    toc()

    return merge(flattened_tree(t),
                 Dict{String, Vector}("mu" => mu, "omega" => omega))
end


function approximate_likelihood{GRADONLY}(approx::NormalALRApprox,
                                          X::SparseMatrixCSC,
                                          ::Type{Val{GRADONLY}}=Val{false})

    m, n = size(X)
    Xt = transpose(X)
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(n-1)
    m_omega = Array{Float32}(n-1)

    # gradient running variances
    v_mu    = Array{Float32}(n-1)
    v_omega = Array{Float32}(n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1

    # cluster transcripts for hierachrical stick breaking
    refidx = approx.reference_idx == -1 ? n :
             approx.reference_idx == 0 ? rand(1:n) : approx.reference_idx
    t = ALRTransform(refidx)

    # standard normal values
    zs = Array{Float32}(n-1)

    # destandardized normal random numbers
    ys = Array{Float64}(n-1)

    # ys transformed to simplex using ILR
    xs = Array{Float32}(n)

    mu = fill(0.0f0, n-1)
    omega = fill(log(0.1f0), n-1)

    sigma = Array{Float32}(n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(n-1)
    omega_grad = Array{Float32}(n-1)
    sigma_grad = Array{Float32}(n-1)
    y_grad = Array{Float32}(n-1)
    x_grad = Array{Float32}(n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0

    tic()
    prog = Progress(LIKAP_NUM_STEPS, 0.25, "Optimizing ", 60)
    for step_num in 1:LIKAP_NUM_STEPS
        learning_rate = adam_learning_rate(step_num - 1)

        elbo = 0.0
        fill!(mu_grad, 0.0f0)
        fill!(omega_grad, 0.0f0)

        for i in 1:n-1
            sigma[i] = exp(omega[i])
        end

        eps = 1e-10

        for _ in 1:LIKAP_NUM_MC_SAMPLES
            fill!(x_grad, 0.0f0)
            fill!(y_grad, 0.0f0)
            fill!(sigma_grad, 0.0f0)

            for i in 1:n-1
                zs[i] = randn(Float32)
                ys[i] = mu[i] + sigma[i] * zs[i]
            end

            alr_ladj = alr_transform!(t, ys, xs, Val{GRADONLY})                     # y -> x
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val{GRADONLY})
            elbo += lp + alr_ladj

            alr_transform_gradients!(t, ys, xs, y_grad, x_grad)

            for i in 1:n-1
                mu_grad[i]    += y_grad[i]
                sigma_grad[i] = zs[i] * y_grad[i]
            end

            # adjust for log transform and accumulate
            for i in 1:n-1
                omega_grad[i] += sigma[i] * sigma_grad[i]
            end
        end

        for i in 1:n-1
            mu_grad[i]    /= LIKAP_NUM_MC_SAMPLES
            omega_grad[i] /= LIKAP_NUM_MC_SAMPLES
            # omega_grad[i] += 1 # entropy gradient
        end

        elbo /= LIKAP_NUM_MC_SAMPLES # get estimated expectation over mc samples
        # elbo += normal_entropy(sigma)

        @show elbo
        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)

        next!(prog)
    end

    toc()

    return Dict{String, Vector}("mu" => mu, "omega" => omega,
                                "refidx" => [t.refidx])

end
