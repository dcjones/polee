
# Alternative likelihood approximations that exist just for the purpose of
# comparison. May be be removed at some point in the future.


"""
Logistic-normal distribution.
"""
struct LogisticNormalApprox <: LikelihoodApproximation end


"""
Hierarchical stick breaking with Kumaraswamy distributed balances.
"""
struct KumaraswamyPTTApprox <: LikelihoodApproximation
    treemethod::Symbol # [one of :sequential, :random, :clustered]
end
KumaraswamyPTTApprox() = KumaraswamyPTTApprox(:clustered)


"""
Hierarchical stick breaking with Logit-Normal distributed balances.
"""
struct LogitNormalPTTApprox <: LikelihoodApproximation
    treemethod::Symbol # [one of :sequential, :random, :clustered]
end
LogitNormalPTTApprox() = LogitNormalPTTApprox(:clustered)

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


function approximate_likelihood(::LogisticNormalApprox, sample::RNASeqSample,
                                ::Val{gradonly}=Val(true)) where {gradonly}
    X = sample.X
    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(undef, n-1)
    m_omega = Array{Float32}(undef, n-1)

    # gradient running variances
    v_mu    = Array{Float32}(undef, n-1)
    v_omega = Array{Float32}(undef, n-1)

    # step size clamp
    ss_max_mu_step    = 2e-2
    ss_max_omega_step = 2e-2

    # Normal distributed values
    zs = Array{Float32}(undef, n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(undef, n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(undef, n)

    mu = fill(0.0f0, n-1)
    omega = fill(0.1f0, n-1)

    # exp(omega)
    sigma = Array{Float32}(undef, n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(undef, n-1)
    omega_grad = Array{Float32}(undef, n-1)
    sigma_grad = Array{Float32}(undef, n-1)
    y_grad = Array{Float32}(undef, n-1)
    x_grad = Array{Float64}(undef, n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

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
                                X, Xt, xs, x_grad, Val(gradonly))

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

        # @show elbo

        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)

        next!(prog)
    end

    return Dict(
        "mu" => mu,
        "omega" => omega)
end




function approximate_likelihood(approx::LogitNormalPTTApprox,
                                sample::RNASeqSample,
                                ::Val{gradonly}=Val(false)) where {gradonly}
    X = sample.X
    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(undef, n-1)
    m_omega = Array{Float32}(undef, n-1)

    # gradient running variances
    v_mu    = Array{Float32}(undef, n-1)
    v_omega = Array{Float32}(undef, n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1

    # cluster transcripts for hierachrical stick breaking
    t = PolyaTreeTransform(X, approx.treemethod)

    # Unifom distributed values
    zs = Array{Float32}(undef, n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(undef, n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(undef, n)

    inverse_transform!(t, fill(1.0f0/n, n), ys)
    mu    = fill(0.0f0, n-1)
    omega = fill(log(0.1f0), n-1)

    # exp(omega)
    sigma = Array{Float32}(undef, n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(undef, n-1)
    omega_grad = Array{Float32}(undef, n-1)
    sigma_grad = Array{Float32}(undef, n-1)
    y_grad = Array{Float32}(undef, n-1)
    x_grad = Array{Float64}(undef, n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

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

            ln_ladj = logit_normal_transform!(mu, sigma, zs, ys, Val(!gradonly))
            ys = clamp!(ys, eps, 1 - eps)

            hsb_ladj = transform!(t, ys, xs, Val(!gradonly))
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val(gradonly))
            elbo = lp + ln_ladj + hsb_ladj

            transform_gradients!(t, ys, y_grad, x_grad)
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

    return Dict{String, Vector}(
        "node_parent_idxs" => t.index[4,:],
        "node_js"          => t.index[1,:],
        "mu" => mu, "omega" => omega)
end


function approximate_likelihood(approx::KumaraswamyPTTApprox,
                                sample::RNASeqSample,
                                ::Val{gradonly}=Val(false)) where {gradonly}
    X = sample.X
    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # gradient running mean
    m_α = Array{Float32}(undef, n-1)
    m_β = Array{Float32}(undef, n-1)

    # gradient running variances
    v_α = Array{Float32}(undef, n-1)
    v_β = Array{Float32}(undef, n-1)

    # step size clamp
    ss_max_α_step = 1e-1
    ss_max_β_step = 1e-1

    # cluster transcripts for hierachrical stick breaking
    m, n = size(X)
    if approx.treemethod == :clustered
        root = hclust(X)
        nodes = order_nodes(root, n)
    elseif approx.treemethod == :random
        nodes = rand_tree_nodes(n)
    elseif approx.treemethod == :sequential
        nodes = rand_list_nodes(n)
    else
        error("$(approx.treemethod) is not a supported Polya tree transform heuristic")
    end
    t = PolyaTreeTransform(nodes)

    # Unifom distributed values
    zs = Array{Float32}(undef, n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(undef, n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(undef, n)

    # log transformed kumaraswamy parameters
    αs = zeros(Float32, n-1)
    βs = zeros(Float32, n-1)

    as = Array{Float32}(undef, n-1) # exp(αs)
    bs = Array{Float32}(undef, n-1) # exp(βs)

    # various intermediate gradients
    α_grad = Array{Float32}(undef, n-1)
    β_grad = Array{Float32}(undef, n-1)
    a_grad = Array{Float32}(undef, n-1)
    b_grad = Array{Float32}(undef, n-1)
    y_grad = Array{Float32}(undef, n-1)
    x_grad = Array{Float64}(undef, n)
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
    k = 1
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

            kum_ladj = kumaraswamy_transform!(as, bs, zs, ys, work, Val(!gradonly))
            ys = clamp!(ys, LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)

            hsb_ladj = transform!(t, ys, xs, Val(!gradonly))
            xs = clamp!(xs, LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val(gradonly))
            elbo = lp + kum_ladj + hsb_ladj

            transform_gradients!(t, ys, y_grad, x_grad)
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

        # # TODO: This loop really shouldn't be allocating anything but on julia
        # # 1.1.0 it does, and I can't figure out why, so I have to periodically
        # # run the gc to make sure I don't run out of memory.
        # gc()

        next!(prog)
    end

    return Dict{String, Vector}(
        "node_parent_idxs" => t.index[4,:],
        "node_js"          => t.index[1,:],
        "alpha" => αs, "beta" => βs)
end




function approximate_likelihood(approx::NormalILRApprox,
                                sample::RNASeqSample,
                                ::Val{gradonly}=Val(false)) where {gradonly}
    X = sample.X
    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(undef, n-1)
    m_omega = Array{Float32}(undef, n-1)

    # gradient running variances
    v_mu    = Array{Float32}(undef, n-1)
    v_omega = Array{Float32}(undef, n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1

    # cluster transcripts for hierachrical stick breaking
    t = ILRTransform(X, approx.treemethod)

    # standard normal values
    zs = Array{Float32}(undef, n-1)

    # destandardized normal random numbers
    ys = Array{Float64}(undef, n-1)

    # ys transformed to simplex using ILR
    xs = Array{Float32}(undef, n)

    mu = fill(0.0f0, n-1)
    omega = fill(log(0.1f0), n-1)

    sigma = Array{Float32}(undef, n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(undef, n-1)
    omega_grad = Array{Float32}(undef, n-1)
    sigma_grad = Array{Float32}(undef, n-1)
    y_grad = Array{Float32}(undef, n-1)
    x_grad = Array{Float32}(undef, n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0

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

            ilr_ladj = ilr_transform!(t, ys, xs, Val(!gradonly))
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val(gradonly))
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

        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)

        next!(prog)
    end

    return merge(flattened_tree(t),
                 Dict{String, Vector}("mu" => mu, "omega" => omega))
end


function approximate_likelihood(approx::NormalALRApprox,
                                sample::RNASeqSample,
                                ::Val{gradonly}=Val(false)) where {gradonly}

    X = sample.X
    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(undef, n-1)
    m_omega = Array{Float32}(undef, n-1)

    # gradient running variances
    v_mu    = Array{Float32}(undef, n-1)
    v_omega = Array{Float32}(undef, n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1

    # cluster transcripts for hierachrical stick breaking
    refidx = approx.reference_idx == -1 ? n :
             approx.reference_idx == 0 ? rand(1:n) : approx.reference_idx
    t = ALRTransform(refidx)

    # standard normal values
    zs = Array{Float32}(undef, n-1)

    # destandardized normal random numbers
    ys = Array{Float64}(undef, n-1)

    # ys transformed to simplex using ILR
    xs = Array{Float32}(undef, n)

    mu = fill(0.0f0, n-1)
    omega = fill(log(0.1f0), n-1)

    sigma = Array{Float32}(undef, n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(undef, n-1)
    omega_grad = Array{Float32}(undef, n-1)
    sigma_grad = Array{Float32}(undef, n-1)
    y_grad = Array{Float32}(undef, n-1)
    x_grad = Array{Float32}(undef, n)
    work   = zeros(Float32, n-1) # used by kumaraswamy_transform!

    elbo = 0.0

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

            alr_ladj = alr_transform!(t, ys, xs, Val(!gradonly))
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val(gradonly))
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

        @assert isfinite(elbo)

        adam_update_mv!(m_mu, v_mu, mu_grad, step_num)
        adam_update_mv!(m_omega, v_omega, omega_grad, step_num)

        adam_update_params!(mu, m_mu, v_mu, learning_rate, step_num, ss_max_mu_step)
        adam_update_params!(omega, m_omega, v_omega, learning_rate, step_num, ss_max_omega_step)

        next!(prog)
    end

    return Dict{String, Vector}("mu" => mu, "omega" => omega,
                                "refidx" => [t.refidx])

end
