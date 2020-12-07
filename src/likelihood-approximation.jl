

abstract type LikelihoodApproximation end

"""
Optimize point estimates using PTT transformation.
"""
struct OptimizePTTApprox <: LikelihoodApproximation end

"""
Polya tree transform with logit skew-normal distribution. Default likelihood
approximation used by polee.
"""
struct LogitSkewNormalPTTApprox <: LikelihoodApproximation
    treemethod::Symbol # [one of :sequential, :random, :cluster]
end
LogitSkewNormalPTTApprox() = LogitSkewNormalPTTApprox(:cluster)


"""
Roughly optimize likelihood by climbing the gradients using a ptt transform.
"""
function optimize_likelihood(sample::RNASeqSample)
    return approximate_likelihood(OptimizePTTApprox(), sample)["x"]
end


"""
Approximate the likelihood function of an RNA-Seq sample and write the result
to a file.
"""
function approximate_likelihood(approximation::LikelihoodApproximation,
                                sample::RNASeqSample, output_filename::String;
                                gene_noninformative::Bool=false,
                                use_efflen_jacobian::Bool=true,
                                tree_topology_input_filename=nothing,
                                tree_topology_output_filename=nothing)
    @tic()
    if !isa(approximation, LogitSkewNormalPTTApprox)
        @warn "Using alternative approximation. Some features not supported."
        params = approximate_likelihood(approximation, sample)
    else
        params = approximate_likelihood(
            approximation, sample, gene_noninformative=gene_noninformative,
            use_efflen_jacobian=use_efflen_jacobian,
            tree_topology_input_filename=tree_topology_input_filename,
            tree_topology_output_filename=tree_topology_output_filename)
    end
    @toc("Approximating likelihood")

    write_approximation(
        output_filename, sample.m, sample.n, sample.effective_lengths,
        params, typeof(approximation),
        sample.transcript_metadata.filename,
        base64encode(sample.transcript_metadata.gffhash),
        sample.sequences_filename,
        base64encode(sample.sequences_file_hash))
end


function write_approximation(
        output_filename::String, m::Int, n::Int, efflens,
        params::Dict, approx_type, gfffilename, gffhash, fafilename, fahash)

    h5open(output_filename, "w") do out
        out["n"] = n
        out["m"] = m
        out["effective_lengths"] = efflens

        for (key, val) in params
            out[key] = val
        end

        g = g_create(out, "metadata")
        # incremented whenever the format changes
        attrs(g)["version"]       = PREPARED_SAMPLE_FORMAT_VERSION
        attrs(g)["approximation"] = string(approx_type)
        attrs(g)["gfffilename"] = gfffilename
        attrs(g)["gffhash"]     = gffhash
        attrs(g)["fafilename"] = fafilename
        attrs(g)["fahash"]     = fahash
        attrs(g)["date"] = string(now())
        attrs(g)["args"] = join(ARGS, ' ')
        # TODO: some other things we might write:
        #   - polee version
    end
end



"""
Throw an error if the prepared sample was generated using an incompatible version.
"""
function check_prepared_sample_version(metadata)
    if !exists(attrs(metadata), "version") ||
        read(attrs(metadata)["version"]) != PREPARED_SAMPLE_FORMAT_VERSION
        error(string("Prepared sample $(filename) was generated using a ",
                        read(attrs(metadata)["version"]) < PREPARED_SAMPLE_FORMAT_VERSION ? "older" : "newer",
                        " version of the software."))
    end
end


"""
Compute ADAM learning rate for the step_num iteration.
"""
function adam_learning_rate(step_num)
    return max(ADAM_MIN_LEARNING_RATE,
        ADAM_INITIAL_LEARNING_RATE * exp(-ADAM_LEARNING_RATE_DECAY * step_num))
end


"""
Compute ADAM momentum/velocity values.
"""
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


"""
Update parameter estimates using ADAM.
"""
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


function approximate_likelihood(
        ::OptimizePTTApprox, sample::RNASeqSample,
        ::Val{gradonly}=Val(true)) where {gradonly}
    X = sample.X
    efflens = sample.effective_lengths

    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # cluster transcripts for hierachrical stick breaking
    t = PolyaTreeTransform(X, :sequential)

    m_z = Array{Float32}(undef, n-1)
    v_z = Array{Float32}(undef, n-1)

    ss_max_z_step = 1e-1

    zs = Array{Float32}(undef, n-1)

    # logistic transformed zs values
    ys = Array{Float64}(undef, n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(undef, n)

    # effective length transformed xs
    xls = Array{Float32}(undef, n)

    z_grad = Array{Float64}(undef, n-1)
    y_grad = Array{Float64}(undef, n-1)
    x_grad = Array{Float64}(undef, n)

    # initial values for z
    fill!(xs, 1.0f0/n)
    inverse_transform!(t, xs, ys)
    for i in 1:n-1
        zs[i] = logit(ys[i])
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

        hsb_ladj = transform!(t, ys, xs, Val(!gradonly))
        xs = clamp!(xs, eps, 1 - eps)

        log_likelihood(model.frag_probs, model.log_frag_probs,
                       X, Xt, xs, x_grad, Val(gradonly))
        effective_length_jacobian_adjustment!(efflens, xs, xls, x_grad)

        transform_gradients_no_ladj!(t, ys, y_grad, x_grad)

        for i in 1:n-1
            dy_dz = ys[i] * (1 - ys[i])
            z_grad[i] = dy_dz * y_grad[i]

            # log jacobian gradient
            # expz = exp(zs[i])
            # z_grad[i] += (1 - expz) / (1 + expz)

            # TODO:
            # equivalent to:
            # z_grad[i] += 1 - 2*ys[i]
            # but both of these version cause in incorrect mode to be found.
            # this seems like it may be at the root of our issue, but I really
            # can't see why this is happening.

            # I't possible that the derivative just gets very small and too
            # flat to effectively climb, but nevertheless it leads to a mode
            # that is way off the mark.
        end

        adam_update_mv!(m_z, v_z, z_grad, step_num)
        adam_update_params!(zs, m_z, v_z, learning_rate, step_num, ss_max_z_step)

        next!(prog)
    end

    for i in 1:n-1
        ys[i] = logistic(zs[i])
    end
    hsb_ladj = transform!(t, ys, xs, Val(!gradonly))
    xs = clamp!(xs, eps, 1 - eps)
    return Dict("x" => xs)
end


"""
Version of logit skew normal approximation using factored likelihood.
"""
function approximate_likelihood(
        approx::LogitSkewNormalPTTApprox, t::PolyaTreeTransform,
        X::SparseMatrixCSC, ks::Vector, efflens::Vector,
        output_filename::String; use_efflen_jacobian::Bool=true)

    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(undef, n-1)
    m_omega = Array{Float32}(undef, n-1)
    m_alpha = Array{Float32}(undef, n-1)

    # gradient running variances
    v_mu    = Array{Float32}(undef, n-1)
    v_omega = Array{Float32}(undef, n-1)
    v_alpha = Array{Float32}(undef, n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1
    ss_max_alpha_step = 2e-2

    # Unifom distributed values
    zs0 = Array{Float32}(undef, n-1)
    zs  = Array{Float32}(undef, n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(undef, n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(undef, n)

    # effective length transformed xs
    xls = Array{Float32}(undef, n)

    inverse_transform!(t, fill(1.0f0/n, n), ys)
    mu = Array{Float32}(undef, n-1)
    map!(logit, mu, ys)

    omega = fill(log(0.1f0), n-1)
    alpha = fill(0.0f0, n-1)

    # exp(omega)
    sigma = Array{Float32}(undef, n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(undef, n-1)
    omega_grad = Array{Float32}(undef, n-1)
    sigma_grad = Array{Float32}(undef, n-1)
    alpha_grad = Array{Float32}(undef, n-1)
    y_grad = Array{Float32}(undef, n-1)
    x_grad = Array{Float64}(undef, n)
    xl_grad = Array{Float64}(undef, n)
    z_grad = Array{Float32}(undef, n-1)

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

            skew_ladj = sinh_asinh_transform!(alpha, zs0, zs, Val(false))
            ln_ladj = logit_normal_transform!(mu, sigma, zs, ys, Val(false))
            ys = clamp!(ys, eps, 1 - eps)

            hsb_ladj = transform!(t, ys, xs, Val(false))
            xs = clamp!(xs, eps, 1 - eps)

            lp = factored_log_likelihood(
                model.frag_probs, model.log_frag_probs,
                X, Xt, ks, xs, x_grad, Val(true))

            if use_efflen_jacobian
                lp += effective_length_jacobian_adjustment!(efflens, xs, xls, x_grad)
            end

            elbo = lp + skew_ladj + ln_ladj + hsb_ladj

            transform_gradients!(t, ys, y_grad, x_grad)
            logit_normal_transform_gradients!(zs, ys, mu, sigma, y_grad, z_grad, mu_grad, sigma_grad)
            sinh_asinh_transform_gradients!(zs0, alpha, z_grad, alpha_grad)

            # adjust for log transform and accumulate
            for i in 1:n-1
                omega_grad[i] += sigma[i] * sigma_grad[i]
            end
        end

        all_finite = true
        for i in 1:n-1
            mu_grad[i]    /= LIKAP_NUM_MC_SAMPLES
            omega_grad[i] /= LIKAP_NUM_MC_SAMPLES
            alpha_grad[i] /= LIKAP_NUM_MC_SAMPLES
            all_finite &= isfinite(mu_grad[i]) && isfinite(omega_grad[i]) && isfinite(alpha_grad[i])
        end
        @assert all_finite

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

    params = Dict{String, Vector}(
        "mu" => mu, "omega" => omega, "alpha" => alpha)

    write_approximation(
        output_filename, m, n, efflens, params, typeof(approx), "", "", "", "")
end


function approximate_likelihood(approx::LogitSkewNormalPTTApprox,
                                sample::RNASeqSample,
                                ::Val{gradonly}=Val(true);
                                tree_topology_input_filename=nothing,
                                tree_topology_output_filename=nothing,
                                gene_noninformative::Bool=false,
                                use_efflen_jacobian::Bool=true) where {gradonly}
    X = sample.X

    efflens = sample.effective_lengths

    m, n = size(X)
    Xt = SparseMatrixCSC(transpose(X))
    model = Model(m, n)

    # gradient running mean
    m_mu    = Array{Float32}(undef, n-1)
    m_omega = Array{Float32}(undef, n-1)
    m_alpha = Array{Float32}(undef, n-1)

    # gradient running variances
    v_mu    = Array{Float32}(undef, n-1)
    v_omega = Array{Float32}(undef, n-1)
    v_alpha = Array{Float32}(undef, n-1)

    # step size clamp
    ss_max_mu_step    = 2e-1
    ss_max_omega_step = 2e-1
    ss_max_alpha_step = 2e-2

    # cluster transcripts for hierachrical stick breaking
    tree_nodes_ref = Vector{HClustNode}[]

    if tree_topology_input_filename !== nothing
        input = h5open(tree_topology_input_filename)
        t = PolyaTreeTransform(
            read(input["node_parent_idxs"]),
            read(input["node_js"]))
        close(input)
    else
        t = PolyaTreeTransform(X, approx.treemethod, tree_nodes_ref)
    end

    # Unifom distributed values
    zs0 = Array{Float32}(undef, n-1)
    zs  = Array{Float32}(undef, n-1)

    # zs transformed to Kumaraswamy distributed values
    ys = Array{Float64}(undef, n-1)

    # ys transformed by hierarchical stick breaking
    xs = Array{Float32}(undef, n)

    # effective length transformed xs
    xls = Array{Float32}(undef, n)

    inverse_transform!(t, fill(1.0f0/n, n), ys)
    mu = Array{Float32}(undef, n-1)
    map!(logit, mu, ys)

    omega = fill(log(0.1f0), n-1)
    alpha = fill(0.0f0, n-1)

    # exp(omega)
    sigma = Array{Float32}(undef, n-1)

    # various intermediate gradients
    mu_grad    = Array{Float32}(undef, n-1)
    omega_grad = Array{Float32}(undef, n-1)
    sigma_grad = Array{Float32}(undef, n-1)
    alpha_grad = Array{Float32}(undef, n-1)
    y_grad = Array{Float32}(undef, n-1)
    x_grad = Array{Float64}(undef, n)
    xl_grad = Array{Float64}(undef, n)
    z_grad = Array{Float32}(undef, n-1)

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

    # Map gene_id to vectors of transcript indexes
    gene_transcripts = Dict{String, Vector{Int}}()
    if gene_noninformative
        for (i, t) in enumerate(sample.ts)
            tid = t.metadata.name
            if haskey(sample.transcript_metadata.gene_id, tid)
                gid = sample.transcript_metadata.gene_id[tid]
                if !haskey(gene_transcripts, gid)
                    gene_transcripts[gid] = Int[]
                end
                push!(gene_transcripts[gid], i)
            end
        end

        if isempty(gene_transcripts)
            @warn "'--gene-noninformative' used, but no gene information available"
            gene_noninformative = false
        end
    end

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

            skew_ladj = sinh_asinh_transform!(alpha, zs0, zs, Val(!gradonly))
            ln_ladj = logit_normal_transform!(mu, sigma, zs, ys, Val(!gradonly))
            ys = clamp!(ys, eps, 1 - eps)

            hsb_ladj = transform!(t, ys, xs, Val(!gradonly))
            xs = clamp!(xs, eps, 1 - eps)

            lp = log_likelihood(model.frag_probs, model.log_frag_probs,
                                X, Xt, xs, x_grad, Val(gradonly))

            if use_efflen_jacobian
                lp += effective_length_jacobian_adjustment!(efflens, xs, xls, x_grad)
            end

            if gene_noninformative
                lp += gene_noninformative_prior!(
                    efflens, xls, xl_grad, xs, x_grad, gene_transcripts)
            end

            elbo = lp + skew_ladj + ln_ladj + hsb_ladj

            transform_gradients!(t, ys, y_grad, x_grad)
            logit_normal_transform_gradients!(zs, ys, mu, sigma, y_grad, z_grad, mu_grad, sigma_grad)
            sinh_asinh_transform_gradients!(zs0, alpha, z_grad, alpha_grad)

            # adjust for log transform and accumulate
            for i in 1:n-1
                omega_grad[i] += sigma[i] * sigma_grad[i]
            end
        end

        all_finite = true
        for i in 1:n-1
            mu_grad[i]    /= LIKAP_NUM_MC_SAMPLES
            omega_grad[i] /= LIKAP_NUM_MC_SAMPLES
            alpha_grad[i] /= LIKAP_NUM_MC_SAMPLES
            all_finite &= isfinite(mu_grad[i]) && isfinite(omega_grad[i]) && isfinite(alpha_grad[i])
        end
        @assert all_finite

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

    # optionally write tree diagnostics
    if tree_topology_output_filename !== nothing && !isempty(tree_nodes_ref)
        ts_names = [t.metadata.name for t in sample.ts]
        ts_gene_names = [
            get(sample.transcript_metadata.gene_name,
                get(sample.transcript_metadata.gene_id, t.metadata.name, ""), "")
            for t in sample.ts]

        nodes = tree_nodes_ref[1]
        node_ids = IdDict{HClustNode, Int}()
        for (i, node) in enumerate(nodes)
            node_ids[node] = i
        end

        open(tree_topology_output_filename, "w") do output
            println(output, "nodes:")
            k = 1

            for (i, node) in enumerate(nodes)
                println(output, "  - id: node", i)
                println(output, "    read_count: ", node.read_count)
                # internal node
                if node.j == 0
                    println(output, "    mu: ", mu[k])
                    println(output, "    sigma: ", exp(omega[k]))
                    println(output, "    alpha: ", alpha[k])
                    println(output, "    child_jaccard: ", node.child_jaccard)
                    println(output, "    left: node", node_ids[node.left_child])
                    println(output, "    right: node", node_ids[node.left_child])
                    k += 1
                else
                    println(output, "    transcript_id: ", ts_names[node.j])
                    println(output, "    gene_name: ", ts_gene_names[node.j])
                end
            end
        end
    end

    params = Dict{String, Vector}(
        "mu" => mu, "omega" => omega, "alpha" => alpha)

    if tree_topology_input_filename === nothing
        params["node_parent_idxs"] = t.index[4,:]
        params["node_js"] = t.index[1,:]
    end

    return params
end

