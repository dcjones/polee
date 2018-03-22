
function tf_print_span(var, label)
    return tf.Print(var, [tf.reduce_min(var), tf.reduce_max(var)], label)
end


function estimate_expression(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_expression(input)
    elseif input.feature == :gene
        return estimate_gene_expression(input)
    elseif input.feature == :splicing
        return estimate_splicing_expression(input)
    else
        error("Expression estimates for $(input.feature) not supported")
    end
end


function estimate_transcript_expression(input::ModelInput, write_results::Bool=true)

    idx = 0
    for (i, t) in enumerate(input.ts)
        if t.metadata.name == "transcript:ENST00000396251"
            idx = i
            break
        end
    end
    @show idx
    @show input.loaded_samples.efflen_values[:, idx]
    @show input.loaded_samples.x0_values[:, idx]
    exit()

    num_samples, n = size(input.loaded_samples.x0_values)

    x0_log = log.(input.loaded_samples.x0_values)

    x, x_sigma_sq, x_mu_param, x_mu, likapprox =
        transcript_quantification_model(input)

    println("Estimating...")

    qx_mu_param = tf.Variable(x0_log)
    qx_softplus_sigma_param = tf.Variable(tf.fill([num_samples, n], -1.0f0))
    qx = edmodels.NormalWithSoftplusScale(loc=qx_mu_param, scale=qx_softplus_sigma_param)

    qx_mu_mu_param = tf.Variable(mean(x0_log, 1)[1,:])
    qx_mu_softplus_sigma_param = tf.Variable(tf.fill([n], -1.0f0))
    qx_mu = edmodels.NormalWithSoftplusScale(loc=qx_mu_mu_param, scale=qx_mu_softplus_sigma_param)

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    inference = ed.KLqp(Dict(x => qx, x_mu => qx_mu, x_sigma_sq => qx_sigma_sq),
                        data=Dict(likapprox => Float32[]))
    optimizer = tf.train[:AdamOptimizer](0.02)
    run_inference(input, inference, 500, optimizer)

    sess = ed.get_session()

    mean_est  = sess[:run](qx_mu_param)
    sigma_est = sess[:run](tf.nn[:softplus](qx_softplus_sigma_param))

    lower_credible = similar(mean_est)
    upper_credible = similar(mean_est)
    for i in 1:size(mean_est, 1)
        for j in 1:size(mean_est, 2)
            dist = Normal(mean_est[i, j], sigma_est[i, j])

            lower_credible[i, j] = quantile(dist, input.credible_interval[1])
            upper_credible[i, j] = quantile(dist, input.credible_interval[2])
        end
    end

    # TODO: this should be a temporary measure until we decide exactly how
    # results should be reported. Probably in sqlite or something.
    if write_results
        write_transcript_expression_csv("transcript-expression-estimates.csv",
                                        input.ts, input.loaded_samples.sample_names,
                                        mean_est, lower_credible, upper_credible)
    end

    # open("efflen.csv", "w") do out
    #     println(out, "transcript_num,efflen")
    #     for (i, efflen) in enumerate(efflens)
    #         println(out, i, ",", efflen)
    #     end
    # end

    return mean_est, sigma_est
end


"""
Estimate transcript expression treating each sample as independent. (As
apposed to using pooled mean and variance parameters as
`estimate_transcript_expression``)
"""
function estimate_simple_transcript_expression(input::ModelInput, write_results::Bool=true)
    num_samples, n = size(input.loaded_samples.x0_values)

    x = edmodels.MultivariateNormalDiag(tf.constant(log(1/n), shape=[num_samples, n]),
                                        tf.constant(10.0, shape=[num_samples, n]))

    likapprox = RNASeqApproxLikelihood(input, x)

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))
    qx_mu_param = tf.Variable(x0_log)
    qx_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0)))
    qx = edmodels.MultivariateNormalDiag(qx_mu_param, qx_sigma_param)
    inference = ed.KLqp(Dict(x => qx),
                        data=Dict(likapprox => Float32[]))
    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 500, optimizer)

    sess = ed.get_session()

    mean_est  = sess[:run](qx_mu_param)
    sigma_est = sess[:run](qx_sigma_param)

    lower_credible = similar(mean_est)
    upper_credible = similar(mean_est)
    for i in 1:size(mean_est, 1)
        for j in 1:size(mean_est, 2)
            dist = Normal(mean_est[i, j], sigma_est[i, j])

            lower_credible[i, j] = quantile(dist, input.credible_interval[1])
            upper_credible[i, j] = quantile(dist, input.credible_interval[2])
        end
    end

    if write_results
        # TODO: option to toggle output type
        # write_transcript_expression_csv("estimates.csv",
        #                                 input.loaded_samples.sample_names,
        #                                 mean_est, lower_credible, upper_credible)

        # TODO: figure out best way to generate credible intervals
        write_simplified_transcript_expression_csv("estimates.csv", input, mean_est)
    end

    return mean_est, sigma_est
end


function write_transcript_expression_csv(output_filename, ts, sample_names,
                                         est, lower_credible, upper_credible)
    num_samples, n = size(est)
    open(output_filename, "w") do output
        println(output, "sample_name,transcript_id,expression,lower_credible,upper_credible")
        for (i, sample_name) in enumerate(sample_names)
            for (j, t) in enumerate(ts)
                @printf(output, "%s,%s,%e,%e,%e\n", sample_name, t.metadata.name,
                        est[i, j], lower_credible[i, j], upper_credible[i, j])
            end
        end
    end
end


function write_simplified_transcript_expression_csv(output_filename,
                                                    input::ModelInput, est)
    num_samples, n = size(est)

    x = exp.(est)
    x ./= sum(x, 1)
    x .*= 1e6

    open(output_filename, "w") do output
        println(output, "sample_name,gene_id,gene_name,transcript_id,tpm")
        for (i, sample_name) in enumerate(input.loaded_samples.sample_names)
            for (j, t) in enumerate(input.ts)
                @printf(output, "%s,%s,%s,%s,%0.4f\n",
                        sample_name, t.metadata.name,
                        input.ts_metadata.gene_id[t.metadata.name],
                        input.ts_metadata.gene_name[t.metadata.name],
                        x[i, j])
            end
        end
    end
end


function estimate_gene_expression(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
        gene_feature_matrix(input.ts, input.ts_metadata)
    num_aux_features = regularize_disjoint_feature_matrix!(gene_idxs, transcript_idxs, n)

    prior_vars, prior_var_approximations =
        model_disjoint_feature_prior(input, gene_idxs, transcript_idxs)

    vars, var_approximations, data =
        model_disjoint_feature_expression(input, gene_idxs, transcript_idxs,
                                          num_features + num_aux_features,
                                          prior_vars[:x_feature])

    merge!(var_approximations, prior_var_approximations)
    merge!(vars, prior_vars)

    inference = ed.KLqp(latent_vars=var_approximations, data=data)
    optimizer = tf.train[:AdamOptimizer](5e-2)
    run_inference(input, inference, 1000, optimizer)

    sess = ed.get_session()
    est  = sess[:run](tf.nn[:softmax](var_approximations[vars[:x_feature]][:mean]()))

    if input.output_format == :csv
        output_filename = isnull(input.output_filename) ? "gene-expression.csv" : get(input.output_filename)
        write_gene_expression_csv(output_filename, input.loaded_samples.sample_names,
                                  gene_ids, gene_names, num_aux_features, est)
    elseif input.output_format == :sqlite3
        error("Sqlite3 output for gene expression is not implemented.")
    end
end


function write_gene_expression_csv(output_filename, sample_names,
                                   gene_ids, gene_names,
                                   num_aux_features, est)
    n = size(est, 2) - num_aux_features
    @assert length(gene_names) == n
    open(output_filename, "w") do output
        println(output, "sample_name,gene_id,gene_name,tpm")
        for (i, sample_name) in enumerate(sample_names)
            for (j, gene_name) in enumerate(gene_names)
                @printf(output, "%s,%s,%s,%e\n", sample_name,
                        gene_ids[j], gene_name, 1e6 * est[i,j])
            end
        end
    end
end


function estimate_splicing_expression(input::ModelInput; write_results::Bool=true)

    num_samples, n = size(input.loaded_samples.x0_values)

    (num_features,
     feature_idxs, feature_transcript_idxs,
     antifeature_idxs, antifeature_transcript_idxs) = splicing_features(input)

    prior_vars, prior_var_approximations = model_nondisjoint_feature_prior(input, num_features)
    vars, var_approximations, data =
        model_nondisjoint_feature_expression(input, num_features,
                                            feature_idxs, feature_transcript_idxs,
                                            antifeature_idxs, antifeature_transcript_idxs,
                                            prior_vars[:x_feature])
    merge!(var_approximations, prior_var_approximations)
    merge!(vars, prior_vars)

    if input.inference == :variational
        inference = ed.KLqp(var_approximations, data=data)

        optimizer = tf.train[:AdamOptimizer](5e-2)
        # inference[:run](n_iter=5000, optimizer=optimizer, logdir="log")
        run_inference(input, inference, 2000, optimizer)

        sess = ed.get_session()
        mean_est  = sess[:run](var_approximations[vars[:x_feature]][:mean]())
        sigma_est  = sess[:run](var_approximations[vars[:x_feature]][:variance]())

        est = similar(mean_est)
        lower_credible = similar(mean_est)
        upper_credible = similar(mean_est)
        for i in 1:size(mean_est, 1)
            for j in 1:size(mean_est, 2)
                dist = Normal(mean_est[i, j], sigma_est[i, j])
                est[i,j ] = logistic(mean_est[i, j])
                lower_credible[i, j] = logistic(quantile(dist, input.credible_interval[1]))
                upper_credible[i, j] = logistic(quantile(dist, input.credible_interval[2]))
            end
        end

        output_filename = isnull(input.output_filename) ?
            "splicing-proportion-estimates.csv" : get(input.output_filename)
        write_splicing_proportions_csv(output_filename, input.loaded_samples.sample_names,
                                       num_features, est,
                                       lower_credible, upper_credible)

        return mean_est
    elseif input.inference == :map

        qx_feature_param = tf.Variable(tf.zeros([num_samples, num_features]))
        qx_feature = edmodels.PointMass(params=qx_feature_param)

        var_maps = Dict(
            vars[:x_feature_mu] =>
                edmodels.PointMass(tf.Variable(tf.zeros([num_features]))),
            vars[:x_feature_log_sigma] =>
                edmodels.PointMass(tf.Variable(tf.zeros([num_features]))),
            vars[:x_feature] => qx_feature,
            vars[:x] =>
                edmodels.PointMass(tf.Variable(tf.fill([num_samples, n], log(1.0f0/n)))),
            vars[:x_sigma_sq] =>
                edmodels.PointMass(tf.Variable(tf.fill([n], 0.1))),
            vars[:x_component_mu] =>
                edmodels.PointMass(tf.Variable(var_approximations[vars[:x_component_mu]][:mean]())),
            vars[:x_component] =>
                edmodels.PointMass(tf.Variable(var_approximations[vars[:x_component]][:mean]()))
        )

        inference = ed.MAP(var_maps, data=data)
        optimizer = tf.train[:AdamOptimizer](5e-2)
        # TODO: really neeed a way to set this
        # run_inference(input, inference, 250, optimizer)
        run_inference(input, inference, 50, optimizer)
        sess = ed.get_session()

        est = sess[:run](qx_feature_param)

        # TODO: optionally write results

        # reset_graph()
        return est
    end

end


function splicing_features(input::ModelInput)
    cassette_exons = get_cassette_exons(input.ts)
    # TODO: also include alt acceptor/donor sites and maybe alt UTRs

    # TODO: move this to another function
    db = input.gene_db
    SQLite.execute!(db, "drop table if exists splicing_features")
    SQLite.execute!(db,
        """
        create table splicing_features
        (
            feature_num INT PRIMARY KEY,
            type TEXT,
            seqname TEXT,
            included_first INT,
            included_last INT,
            excluded_first INT,
            excluded_last INT
        )
        """)
    ins_stmt = SQLite.Stmt(db,
        "insert into splicing_features values (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
    SQLite.execute!(db, "begin transaction")
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        SQLite.bind!(ins_stmt, 1, i)
        SQLite.bind!(ins_stmt, 2, "cassette_exon")
        SQLite.bind!(ins_stmt, 3, intron.seqname)
        SQLite.bind!(ins_stmt, 4, flanks.metadata[1])
        SQLite.bind!(ins_stmt, 5, flanks.metadata[2])
        SQLite.bind!(ins_stmt, 6, flanks.first)
        SQLite.bind!(ins_stmt, 7, flanks.last)
        SQLite.execute!(ins_stmt)
    end
    SQLite.execute!(db, "end transaction")

    SQLite.execute!(db, "drop table if exists splicing_feature_including_transcripts")
    SQLite.execute!(db,
        """
        create table splicing_feature_including_transcripts
        (
            feature_num INT KEY,
            transcript_num INT
        )
        """)

    SQLite.execute!(db, "drop table if exists splicing_feature_excluding_transcripts")
    SQLite.execute!(db,
        """
        create table splicing_feature_excluding_transcripts
        (
            feature_num INT KEY,
            transcript_num INT
        )
        """)

    inc_ins_stmt = SQLite.Stmt(db,
        "insert into splicing_feature_including_transcripts values (?1, ?2)")
    exc_ins_stmt = SQLite.Stmt(db,
        "insert into splicing_feature_excluding_transcripts values (?1, ?2)")
    SQLite.execute!(db, "begin transaction")
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        SQLite.bind!(inc_ins_stmt, 1, i)
        SQLite.bind!(exc_ins_stmt, 1, i)

        for id in flanks.metadata[3]
            SQLite.bind!(inc_ins_stmt, 2, id)
            SQLite.execute!(inc_ins_stmt)
        end

        for id in intron.metadata
            SQLite.bind!(exc_ins_stmt, 2, id)
            SQLite.execute!(exc_ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")


    feature_idxs = Int32[]
    feature_transcript_idxs = Int32[]
    antifeature_idxs = Int32[]
    antifeature_transcript_idxs = Int32[]
    num_features = length(cassette_exons)

    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        @assert !isempty(intron.metadata)
        @assert !isempty(flanks.metadata[3])

        for id in flanks.metadata[3]
            push!(feature_idxs, i)
            push!(feature_transcript_idxs, id)
        end

        for id in intron.metadata
            push!(antifeature_idxs, i)
            push!(antifeature_transcript_idxs, id)
        end
    end

    return num_features,
           feature_idxs, feature_transcript_idxs,
           antifeature_idxs, antifeature_transcript_idxs
end


function write_splicing_proportions_csv(output_filename, sample_names,
                                        num_features, est,
                                        lower_credible, upper_credible)
    open(output_filename, "w") do output
        println(output, "sample_name,feature_num,proportion,lower_credible,upper_credible")
        for (i, sample_name) in enumerate(sample_names)
            for j in 1:num_features
                @printf(output, "%s,%d,%e,%e,%e\n", sample_name, j,
                        est[i, j], lower_credible[i, j], upper_credible[i, j])
            end
        end
    end
end

# function estimate_splicing_log_ratio(input::ModelInput)
#     cassette_exons = get_cassette_exons(input.ts)
#     I = Int[]
#     J = Int[]
#     V = Float32[]
#     for (i, (intron, flanks)) in  enumerate(cassette_exons)
#         for id in intron.metadata
#             push!(I, i)
#             push!(J, id)
#             push!(V, 1.0f0)
#         end

#         for id in flanks.metadata[3]
#             push!(I, i)
#             push!(J, id)
#             push!(V, -1.0f0)
#         end
#     end
#     m = length(cassette_exons)

#     F = tf.SparseTensor(indices=cat(2, I-1, J-1), values=V,
#                         dense_shape=[m, length(input.ts)])

#     qy_mu_value, qy_sigma_value =
#         estimate_feature_expression(input.likapprox_data, input.y0, input.sample_factors, F)
# end

function estimate_splicing_log_ratio(input::ModelInput)
    qy_mu_value, qy_sigma_value = estimate_splicing_expression(input)
    num_samples, n = size(qy_mu_value)

    inc_indexes = [i for i in 1:2:n]
    exc_indexes = [i for i in 2:2:n]

    qy_mu_inc_value = qy_mu_value[:,inc_indexes]
    qy_mu_exc_value = qy_mu_value[:,exc_indexes]

    qy_sigma_inc_value = qy_sigma_value[:,inc_indexes]
    qy_sigma_exc_value = qy_sigma_value[:,exc_indexes]

    qy_mu_ratio_value    = qy_mu_inc_value .- qy_mu_exc_value
    qy_sigma_ratio_value = qy_sigma_inc_value +- qy_sigma_exc_value

    @show minimum(qy_mu_inc_value), median(qy_mu_inc_value), maximum(qy_mu_inc_value)

    cassette_exons = get_cassette_exons(input.ts)

    println("EXTREME INC VALUES")
    for i in 1:length(qy_mu_inc_value)
        if qy_mu_inc_value[i] < -500.0
            @show (i, qy_mu_inc_value[i], qy_mu_exc_value[i], qy_sigma_inc_value[i], qy_sigma_exc_value[i])
            @show (length(cassette_exons[i][1].metadata), length(cassette_exons[i][2].metadata[3]))
        end
    end

    println("EXTREME EXC VALUES")
    for i in 1:length(qy_mu_exc_value)
        if qy_mu_exc_value[i] < -500.0
            @show (i, qy_mu_inc_value[i], qy_mu_exc_value[i], qy_sigma_inc_value[i], qy_sigma_exc_value[i])
            @show (length(cassette_exons[i][1].metadata), length(cassette_exons[i][2].metadata[3]))
        end
    end

    # tmp = sort(qy_mu_inc_value[2,:])
    # @show tmp[1:10]
    # @show tmp[end-10:end]

    exit()

    # @show minimum(qy_mu_exc_value), median(qy_mu_exc_value), maximum(qy_mu_exc_value)
    # @show minimum(qy_mu_ratio_value), median(qy_mu_ratio_value), maximum(qy_mu_ratio_value)
    # @show minimum(qy_sigma_ratio_value), median(qy_sigma_ratio_value), maximum(qy_sigma_ratio_value)

    return qy_mu_ratio_value, qy_sigma_ratio_value
end


"""
Build basic transcript quantification model with some shrinkage towards a pooled mean.
"""
function transcript_quantification_model(input::ModelInput, pooled_means::Bool=true)
    num_samples, n = size(input.loaded_samples.x0_values)

    x_mu_mu0 = tf.constant(log(1f0/n), shape=[n])
    x_mu_sigma0 = tf.constant(4.0f0, shape=[n])
    x_mu = edmodels.Normal(loc=x_mu_mu0, scale=x_mu_sigma0)

    # x_mu_mu0a = tf.constant(log(0.01 * 1/n), shape=[n])
    # x_mu_sigma0a = tf.constant(0.5, shape=[n])

    # x_mu_mu0b = tf.constant(log(0.1 * 1/n), shape=[n])
    # x_mu_sigma0b = tf.constant(5.0, shape=[n])

    # x_mu_probs = tf.stack([tf.constant(0.80, shape=[n]),
    #                        tf.constant(0.20, shape=[n])], -1)

    # x_mu = edmodels.Mixture(
    #     cat=tfdist.Categorical(probs=x_mu_probs),
    #     components=[
    #         edmodels.Normal(loc=x_mu_mu0a, scale=x_mu_sigma0a),
    #         edmodels.Normal(loc=x_mu_mu0b, scale=x_mu_sigma0b)])

        # components=[
        #     edmodels.MultivariateNormalDiag(x_mu_mu0a, x_mu_sigma0a),
        #     edmodels.MultivariateNormalDiag(x_mu_mu0b, x_mu_sigma0b)])

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)

    # y: quantification
    x_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                           tf.expand_dims(x_mu, 0))

    # x_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
    #                           tf.expand_dims(x_sigma, 0))

    x = edmodels.Normal(loc=x_mu_param, scale=x_sigma)
    likapprox = RNASeqApproxLikelihood(input, x)

    return x, x_sigma_sq, x_mu_param, x_mu, likapprox



end


"""
Set up prior on feature expression for simple quantification.
"""
function model_disjoint_feature_prior(input::ModelInput, feature_idxs, transcript_idxs)
    num_features = maximum(feature_idxs)

    num_samples, n = size(input.loaded_samples.x0_values)

    x_feature_mu_mu0 = tf.constant(log(1/num_features), shape=[num_features])
    x_feature_mu_sigma0 = tf.constant(4.0, shape=[num_features])
    x_feature_mu = edmodels.Normal(loc=x_feature_mu_mu0, scale=x_feature_mu_sigma0)

    x_feature_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[num_features])
    x_feature_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[num_features])
    x_feature_sigma_sq = edmodels.InverseGamma(x_feature_sigma_alpha0,
                                               x_feature_sigma_beta0)
    x_feature_sigma = tf.sqrt(x_feature_sigma_sq)

    x_feature_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                   tf.expand_dims(x_feature_mu, 0))

    x_feature_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                      tf.expand_dims(x_feature_sigma, 0))

    x_feature = edmodels.Normal(loc=x_feature_mu_param,
                                scale=x_feature_sigma_param)

    # approximations
    # --------------

    # figure out some reasonable initial values
    feature_mu_initial = zeros(Float32, (num_samples, num_features))
    for i in 1:num_samples
        for (j, k) in zip(feature_idxs, transcript_idxs)
            feature_mu_initial[i, j] += input.loaded_samples.x0_values[i, k]
        end
    end
    map!(log, feature_mu_initial, feature_mu_initial)
    feature_mu_initial_mean = reshape(mean(feature_mu_initial, 1), (num_features,))

    qx_feature_mu_mu_param = tf.Variable(feature_mu_initial_mean)
    qx_feature_mu_softplus_sigma_param = tf.Variable(tf.fill([num_features], -1.0f0))
    qx_feature_mu = edmodels.NormalWithSoftplusScale(loc=qx_feature_mu_mu_param,
                                                     scale=qx_feature_mu_softplus_sigma_param)

    qx_feature_sigma_sq_mu_param    = tf.Variable(tf.fill([num_features], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_feature_sigma_sq_sigma_param = tf.Variable(tf.fill([num_features], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_feature_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(loc=qx_feature_sigma_sq_mu_param,
                                                      scale=qx_feature_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    qx_feature_mu_param = tf.Variable(feature_mu_initial)
    qx_feature_softplus_sigma_param = tf.Variable(tf.fill([num_samples, num_features], -1.0f0))
    qx_feature = edmodels.NormalWithSoftplusScale(loc=qx_feature_mu_param,
                                                  scale=qx_feature_softplus_sigma_param)

    prior_var_approximations = Dict(
        x_feature             => qx_feature,
        x_feature_mu          => qx_feature_mu,
        x_feature_sigma_sq    => qx_feature_sigma_sq,
    )

    prior_vars = Dict(
        :x_feature           => x_feature,
        :x_feature_mu        => x_feature_mu,
        :x_feature_sigma_sq  => x_feature_sigma_sq
    )

    return prior_vars, prior_var_approximations
end


function model_disjoint_feature_expression(input::ModelInput, feature_idxs,
                                           transcript_idxs, num_features,
                                           x_feature)

    num_samples, n = size(input.loaded_samples.x0_values)

    # within feature relative expression of feature constituents
    x_constituent_mu_mu0 = tf.constant(0.0, shape=[n])
    x_constituent_mu_sigma0 = tf.constant(4.0, shape=[n])
    x_constituent_mu = edmodels.Normal(loc=x_constituent_mu_mu0,
                                       scale=x_constituent_mu_sigma0)

    x_constituent_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_constituent_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_constituent_sigma_sq = edmodels.InverseGamma(x_constituent_sigma_alpha0,
                                                   x_constituent_sigma_beta0)
    x_constituent_sigma = tf.sqrt(x_constituent_sigma_sq)

    x_constituent_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                       tf.expand_dims(x_constituent_mu, 0))

    x_constituent_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                          tf.expand_dims(x_constituent_sigma, 0))

    x_constituent = edmodels.Normal(loc=x_constituent_mu_param,
                                    scale=x_constituent_sigma_param)

    # transcript expression as a deterministic function of feature expression
    # and constuent relative expression.
    p = sortperm(feature_idxs)
    permute!(feature_idxs, p)
    permute!(transcript_idxs, p)
    x_constituent_indices = Array{Int32}((n, 2))
    for (k, (i, j)) in enumerate(zip(feature_idxs, transcript_idxs))
        x_constituent_indices[k, 1] = i - 1
        x_constituent_indices[k, 2] = j - 1
    end

    xs = []
    for (x_feature_i, x_constituent_i) in zip(tf.unstack(x_feature), tf.unstack(x_constituent))
        x_constituent_matrix = tf.SparseTensor(indices=x_constituent_indices,
                                               values=x_constituent_i,
                                               dense_shape=[num_features, n])
        x_constituent_matrix_softmax = tf.sparse_softmax(x_constituent_matrix)

        x_i = tf.log(tf.sparse_tensor_dense_matmul(x_constituent_matrix_softmax,
                                                   tf.expand_dims(tf.exp(x_feature_i), -1),
                                                   adjoint_a=true))

        push!(xs, x_i)
    end
    x = tf.squeeze(tf.stack(xs), axis=-1)
    likapprox = RNASeqApproxLikelihood(input, x)

    # Inference
    # ---------

    # figure out some reasonable initial values
    feature_mu_initial     = zeros(Float32, (num_samples, num_features))
    constituent_mu_initial = zeros(Float32, (num_samples, n))
    for i in 1:num_samples
        for (j, k) in zip(feature_idxs, transcript_idxs)
            feature_mu_initial[i, j] += input.loaded_samples.x0_values[i, k]
            constituent_mu_initial[i, k] = input.loaded_samples.x0_values[i, k]
        end

        for (j, k) in zip(feature_idxs, transcript_idxs)
            constituent_mu_initial[i, k] /= feature_mu_initial[i, j]
        end
    end

    map!(log, feature_mu_initial, feature_mu_initial)
    map!(log, constituent_mu_initial, constituent_mu_initial)
    feature_mu_initial_mean = reshape(mean(feature_mu_initial, 1), (num_features,))
    constituent_mu_initial_mean = reshape(mean(constituent_mu_initial, 1), (n,))

    qx_constituent_mu_mu_param = tf.Variable(constituent_mu_initial_mean)
    qx_constituent_mu_softplus_sigma_param = tf.Variable(tf.fill([n], -1.0f0))
    qx_constituent_mu = edmodels.NormalWithSoftplusScale(loc=qx_constituent_mu_mu_param,
                                                         scale=qx_constituent_mu_softplus_sigma_param)

    qx_constituent_sigma_sq_mu_param    = tf.Variable(tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_constituent_sigma_sq_sigma_param = tf.Variable(tf.fill([n], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_constituent_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(loc=qx_constituent_sigma_sq_mu_param,
                                                      scale=qx_constituent_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    qx_constituent_mu_param = tf.Variable(constituent_mu_initial)
    qx_constituent_softplus_sigma_param = tf.Variable(tf.fill([num_samples, n], -1.0f0))
    qx_constituent = edmodels.NormalWithSoftplusScale(loc=qx_constituent_mu_param,
                                                      scale=qx_constituent_softplus_sigma_param)

    vars = Dict(
        :x_constituent => x_constituent,
        :x_constituent_mu => x_constituent_mu,
        :x_constituent_sigma_sq => x_constituent_sigma_sq
    )

    var_approximations = Dict(
        x_constituent          => qx_constituent,
        x_constituent_mu       => qx_constituent_mu,
        x_constituent_sigma_sq => qx_constituent_sigma_sq
    )

    data = Dict(likapprox => Float32[])

    return vars, var_approximations, data
end


"""
Find connected components of transcripts where transcripts are connected if
they share a feature, or one has a feature and another an antifeature.
"""
function connected_components_from_features(n, num_features, feature_idxs, feature_transcript_idxs,
                                            antifeature_idxs, antifeature_transcript_idxs)
    components = IntDisjointSets(n + num_features)

    @assert length(feature_idxs) == length(feature_transcript_idxs)
    for (i, j) in zip(feature_transcript_idxs, feature_idxs)
        union!(components, i, n + j)
    end

    @assert length(antifeature_idxs) == length(antifeature_transcript_idxs)
    for (i, j) in zip(antifeature_transcript_idxs, antifeature_idxs)
        union!(components, i, n + j)
    end

    transcripts_with_features = IntSet()
    component_idx_map = Dict{Int, Int}()
    component_idxs = Int32[]
    component_transcript_idxs = Int32[]

    for i in feature_transcript_idxs
        if i ∈ transcripts_with_features
            continue
        end
        push!(transcripts_with_features, i)
        r = find_root(components, i)
        push!(component_transcript_idxs, i)
        push!(component_idxs,
              get!(component_idx_map, r, 1 + length(component_idx_map)))
    end

    for i in antifeature_transcript_idxs
        if i ∈ transcripts_with_features
            continue
        end
        push!(transcripts_with_features, i)
        r = find_root(components, i)
        push!(component_transcript_idxs, i)
        push!(component_idxs,
              get!(component_idx_map, r, 1 + length(component_idx_map)))
    end

    # add a component for every transcript that has no features
    aux_components = 0
    transcripts_without_features = Int[]
    for i in 1:n
        if i ∉ transcripts_with_features
            r = 1 + aux_components + length(component_idx_map)
            push!(component_transcript_idxs, i)
            push!(component_idxs, r)
            push!(transcripts_without_features, i)
            aux_components += 1
        end
    end

    num_components = aux_components + length(component_idx_map)

    return num_components, component_idxs,
           component_transcript_idxs, transcripts_without_features
end


function nondisjoint_feature_initialization(x0, num_components, num_features,
                                            component_idxs, component_transcript_idxs,
                                            feature_idxs, feature_transcript_idxs,
                                            antifeature_idxs, antifeature_transcript_idxs)
    num_samples, n = size(x0)

    # TODO: this would be better with a more informed initial estimate. Not
    # sure what that would be though.
    x_component_mu_initial = fill(log(0.01f0 * 1f0/num_components), (num_samples, num_components))
    x_feature_mu_initial = zeros(Float32, (num_samples, num_features))

    return x_component_mu_initial, x_feature_mu_initial
end


"""
Estimate the expression of non-disjoint features, each relative to an
"anti-features".

Concretely, a feature here is typically an cassette exon, and the
anti-feature an intron skipping that exon. But this could also be
intron-exclusion/inclusion, alternative poly-A sites, etc.

Features here are non-dijoint in the sense that the sense that they don't
necessarily partition the set of transcripts.

Arguments:
  * `input`: model input
  * `num_features`: number of feature/anti-feature pairs.
  * `feature_idxs`, `feature_transcript_idxs`: matched feature and transcript
  *     indexes, respectively.
  * `antifeature_idxs`, `antifeature_transcript_idxs`: matched antifeature and
        transcript indexes, respectively.

"""
function estimate_nondisjoint_feature_expression(input::ModelInput, num_features,
                                                 feature_idxs, feature_transcript_idxs,
                                                 antifeature_idxs, antifeature_transcript_idxs)

    vars, var_approximations, data =
        model_nondisjoint_feature_expression(input, num_features,
                                            feature_idxs, feature_transcript_idxs,
                                            antifeature_idxs, antifeature_transcript_idxs)

    estimate_nondisjoint_feature_expression(vars, var_approximations, data)
end


function estimate_nondisjoint_feature_expression(vars, var_approximations, data)


    return est, lower_credible, upper_credible
end


function model_nondisjoint_feature_prior(input::ModelInput, num_features)
    num_samples, n = size(input.loaded_samples.x0_values)

    x_feature_mu_mu0 = tf.constant(0.0, shape=[num_features])
    x_feature_mu_sigma0 = tf.constant(5.0, shape=[num_features])
    x_feature_mu = edmodels.Normal(loc=x_feature_mu_mu0,
                                   scale=x_feature_mu_sigma0)

    x_feature_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[num_features])
    x_feature_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[num_features])
    x_feature_sigma_sq = edmodels.InverseGamma(
        x_feature_sigma_alpha0, x_feature_sigma_beta0)
    x_feature_sigma = tf.sqrt(x_feature_sigma_sq)

    x_feature_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                   tf.expand_dims(x_feature_mu, 0))

    x_feature_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                      tf.expand_dims(x_feature_sigma, 0))

    x_feature = edmodels.Normal(loc=x_feature_mu_param,
                                scale=x_feature_sigma_param)

    # TODO: more informed initization
    x_feature_mu_initial = zeros(Float32, (num_samples, num_features))
    x_feature_mu_initial_mean = reshape(mean(x_feature_mu_initial, 1), (num_features,))

    qx_feature_mu_mu_param = tf.Variable(x_feature_mu_initial_mean, name="qx_feature_mu_mu")
    qx_feature_mu_softplus_sigma_param = tf.Variable(tf.fill([num_features], -1.0f0), name="qx_feature_mu_softplus_sigma")
    qx_feature_mu = edmodels.NormalWithSoftplusScale(qx_feature_mu_mu_param, qx_feature_mu_softplus_sigma_param)

    qx_feature_sigma_sq_mu_param    = tf.Variable(tf.fill([num_features], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_feature_sigma_sq_sigma_param = tf.Variable(tf.fill([num_features], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_feature_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(loc=qx_feature_sigma_sq_mu_param,
                                                      scale=qx_feature_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    qx_feature_mu_param = tf.Variable(x_feature_mu_initial, name="qx_feature_mu")
    qx_feature_softplus_sigma_param = tf.Variable(tf.fill([num_samples, num_features], -1.0f0), name="qx_feature_sigma")
    qx_feature = edmodels.NormalWithSoftplusScale(qx_feature_mu_param, qx_feature_softplus_sigma_param)

    prior_var_approximations = Dict(
        x_feature             => qx_feature,
        x_feature_mu          => qx_feature_mu,
        x_feature_sigma_sq    => qx_feature_sigma_sq,
    )

    prior_vars = Dict(
        :x_feature           => x_feature,
        :x_feature_mu        => x_feature_mu,
        :x_feature_sigma_sq  => x_feature_sigma_sq
    )

    return prior_vars, prior_var_approximations
end


function model_nondisjoint_feature_expression(input::ModelInput, num_features,
                                              feature_idxs, feature_transcript_idxs,
                                              antifeature_idxs, antifeature_transcript_idxs,
                                              x_feature)
    num_samples, n = size(input.loaded_samples.x0_values)

    p = sortperm(feature_transcript_idxs)
    permute!(feature_transcript_idxs, p)
    permute!(feature_idxs, p)

    p = sortperm(antifeature_transcript_idxs)
    permute!(antifeature_transcript_idxs, p)
    permute!(antifeature_idxs, p)

    num_components, component_idxs, component_transcript_idxs, transcripts_without_features =
        connected_components_from_features(n, num_features, feature_idxs,
                                        feature_transcript_idxs,
                                        antifeature_idxs,
                                        antifeature_transcript_idxs)

    p = sortperm(component_transcript_idxs)
    permute!(component_transcript_idxs, p)
    permute!(component_idxs, p)

    # transcripts_without_features_props = zeros(Float32, n)
    # transcripts_without_features_props[transcripts_without_features] = 1.0f0
    # transcripts_without_features_props_tensor =
    #     tf.expand_dims(tf.constant(transcripts_without_features_props), -1)

    # component expression
    # --------------------

    x_component_mu_mu0 = tf.constant(log(1/num_components), shape=[num_components])
    x_component_mu_sigma0 = tf.constant(5.0, shape=[num_components])
    x_component_mu = edmodels.Normal(loc=x_component_mu_mu0,
                                     scale=x_component_mu_sigma0)
    # x_component_mu_ = tf_print_span(x_component_mu, "x_component_mu span")

    x_component_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[num_components])
    x_component_sigma_beta0  = tf.constant(SIGMA_BETA0, shape=[num_components])
    x_component_sigma_sq = edmodels.InverseGamma(x_component_sigma_alpha0,
                                                 x_component_sigma_beta0)
    x_component_sigma = tf.sqrt(x_component_sigma_sq)
    # x_component_sigma_ = tf_print_span(x_component_sigma, "x_component_sigma span")


    x_component_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                     tf.expand_dims(x_component_mu, 0))

    x_component = edmodels.Normal(loc=x_component_mu_param,
                                  scale=x_component_sigma)

    # x_component_ = tf_print_span(x_component, "x_component span")

    # feature factors
    # ---------------

    # x_feature_ = tf_print_span(x_feature, "x_feature span")

    x_feature_proportions = tf.sigmoid(x_feature)
    x_feature_proportions = tf.clip_by_value(tf.sigmoid(x_feature), 1f-7, 1.0f0 - 1f-7)
    # x_feature_proportions = tf_print_span(x_feature_proportions, "x_feature_proportions span")
    x_antifeature_proportions = tf.ones([num_samples, num_features]) - x_feature_proportions

    x_feature_log_proportions = tf.log(x_feature_proportions)
    x_antifeature_log_proportions = tf.log(x_antifeature_proportions)

    # x_feature_proportions = tf_print_span(x_feature_proportions, "x_feature_proportions span")
    # x_antifeature_proportions = tf_print_span(x_antifeature_proportions, "x_antifeature_proportions span")


    features_matrix_indices = Array{Int32}((length(feature_idxs), 2))
    for (k, (i, j)) in enumerate(zip(feature_transcript_idxs, feature_idxs))
        features_matrix_indices[k, 1] = i - 1
        features_matrix_indices[k, 2] = j - 1
    end
    features = tf.SparseTensor(indices=features_matrix_indices,
                               values=tf.ones(length(feature_idxs)),
                               dense_shape=[n, num_features])

    antifeatures_matrix_indices = Array{Int32}((length(antifeature_idxs), 2))
    for (k, (i, j)) in enumerate(zip(antifeature_transcript_idxs, antifeature_idxs))
        antifeatures_matrix_indices[k, 1] = i - 1
        antifeatures_matrix_indices[k, 2] = j - 1
    end
    antifeatures = tf.SparseTensor(indices=antifeatures_matrix_indices,
                                   values=tf.ones(length(antifeature_idxs)),
                                   dense_shape=[n, num_features])

    component_matrix_indices = Array{Int32}((length(component_transcript_idxs), 2))
    for (k, (i, j)) in enumerate(zip(component_transcript_idxs, component_idxs))
        component_matrix_indices[k, 1] = i - 1
        component_matrix_indices[k, 2] = j - 1
    end
    components = tf.SparseTensor(indices=component_matrix_indices,
                                 values=tf.ones(length(component_transcript_idxs)),
                                 dense_shape=[n, num_components])

    # @show length(feature_transcript_idxs)
    # @show length(feature_idxs)
    # @show length(antifeature_transcript_idxs)
    # @show length(antifeature_idxs)
    # @show length(component_transcript_idxs)
    # @show length(component_idxs)
    # @show x_feature_log_proportions
    # @show x_antifeature_log_proportions
    # @show x_component
    # exit()

    xs = []
    for (x_feature_log_proportions_i, x_antifeature_log_proportions_i, x_component_i) in
            zip(tf.unstack(x_feature_log_proportions),
                tf.unstack(x_antifeature_log_proportions),
                tf.unstack(x_component))

        transcript_feature_proportions_i =
            tf.sparse_tensor_dense_matmul(features,
                                          tf.expand_dims(x_feature_log_proportions_i, -1))

        # transcript_feature_proportions_i =
        #     tf.Print(transcript_feature_proportions_i,
        #         [tf.gather(tf.squeeze(transcript_feature_proportions_i, axis=-1), idxs_oi)], "FEAT PROP", summarize=4)

        transcript_antifeature_proportions_i =
            tf.sparse_tensor_dense_matmul(antifeatures,
                                          tf.expand_dims(x_antifeature_log_proportions_i, -1))

        # transcript_antifeature_proportions_i =
        #     tf.Print(transcript_antifeature_proportions_i,
        #         [tf.gather(tf.squeeze(transcript_antifeature_proportions_i, axis=-1), idxs_oi)], "ANTIFEAT PROP", summarize=4)

        transcript_component_expression_i =
            tf.sparse_tensor_dense_matmul(components,
                                          tf.expand_dims(x_component_i, -1))
        # transcript_component_expression_i =
        #     tf.sparse_tensor_dense_matmul(components,
        #                                   tf.expand_dims(tf.ones(num_components), -1))

        # transcript_component_expression_i =
        #     tf.Print(transcript_component_expression_i,
        #         [tf.gather(tf.squeeze(transcript_component_expression_i, axis=-1), idxs_oi)], "COMPONENT EX", summarize=4)

        # transcripts_without_features_props_tensor =
        #     tf.Print(transcripts_without_features_props_tensor,
        #         [tf.gather(tf.squeeze(transcripts_without_features_props_tensor, axis=-1), idxs_oi)], "NON FEATURE EX", summarize=4)

        x_i =
            transcript_component_expression_i +
            transcript_feature_proportions_i +
            transcript_antifeature_proportions_i

        push!(xs, x_i)
    end

    x_mu = tf.squeeze(tf.stack(xs), axis=-1)

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0  = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0,
                                       x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)

    # x_sigma = tf_print_span(x_sigma, "x_sigma span")

    x = edmodels.Normal(loc=x_mu, scale=x_sigma)
    # x = edmodels.Normal(loc=x_mu, scale=[0.1f0])
    # x_ = tf.Print(x, [tf.nn[:moments](x, axes=Any[0])], "x moments")

    x_ = x
    # x_ = tf.Print(x_, [tf.reduce_min(x_, axis=1)], "x min 1")
    # x_ = tf.Print(x_, [tf.reduce_max(x_, axis=1)], "x max 1")

    # x_ = tf.Print(x_, [tf.reduce_min(x_, axis=0)], "x min 0", summarize=10)
    # x_ = tf.Print(x_, [tf.reduce_max(x_, axis=0)], "x max 0", summarize=10)
    # x_ = tf.Print(x_, [tf.reduce_mean(x_, axis=0)], "x mean 0", summarize=10)

    likapprox = RNASeqApproxLikelihood(input, x_)

    # Inference
    # ---------

    # variables:
    # - x_component_mu (normal)
    # - x_component_sigma_sq (log normal)
    # - x_component (normal)
    # - x_sigma_sq (log normal)
    # - x (normal)


    x_component_mu_initial, x_feature_mu_initial =
        nondisjoint_feature_initialization(input.loaded_samples.x0_values,
                                           num_components, num_features,
                                           component_idxs, component_transcript_idxs,
                                           feature_idxs, feature_transcript_idxs,
                                           antifeature_idxs, antifeature_transcript_idxs)

    x_component_mu_initial_mean = reshape(mean(x_component_mu_initial, 1), (num_components,))

    qx_component_mu_mu_param = tf.Variable(x_component_mu_initial_mean, name="qx_component_mu_mu_param")
    qx_component_mu_softplus_sigma_param = tf.Variable(tf.fill([num_components], -1.0f0), name="qx_component_mu_sigma_param")
    qx_component_mu = edmodels.NormalWithSoftplusScale(
        loc=qx_component_mu_mu_param, scale=qx_component_mu_softplus_sigma_param,
        name="qx_component_mu")

    qx_component_sigma_sq_mu_param =
        tf.Variable(tf.fill([num_components], -2.0f0),
                    name="qx_component_sigma_sq_mu_param")
    qx_component_sigma_sq_sigma_param =
        tf.Variable(tf.fill([num_components], -1.0f0),
                    name="qx_component_sigma_sq_sigma_param")
    qx_component_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(
            qx_component_sigma_sq_mu_param, qx_component_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="qx_component_sigma_sq")

    qx_component_mu_param = tf.Variable(x_component_mu_initial, name="qx_component_mu_param")
    qx_component_softplus_sigma_param =
        tf.Variable(tf.fill([num_samples, num_components], -1.0f0), name="qx_component_softplus_sigma_param")
    qx_component = edmodels.NormalWithSoftplusScale(
        loc=qx_component_mu_param, scale=qx_component_softplus_sigma_param,
        name="qx_component")

    qx_sigma_sq_mu_param = tf.Variable(tf.fill([n], -3.0f0), name="qx_err_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], -4.0f0), name="qx_err_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(
            qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="qx_sigma_sq")

    # x0_log = log.(input.loaded_samples.x0_values)
    # qx_mu_param = tf.Variable(x0_log, name="qx_mu_param")
    qx_mu_param = tf.Variable(tf.fill([num_samples, n], log(1.0f0/n)), name="qx_mu_param")
    qx_sigma_param = tf.Variable(tf.fill([num_samples, n], -1.0f0), name="qx_sigma_param")
    qx = edmodels.NormalWithSoftplusScale(loc=qx_mu_param, scale=qx_sigma_param, name="qx")

    vars = Dict(
        :x                     => x,
        :x_sigma_sq            => x_sigma_sq,
        :x_component_mu        => x_component_mu,
        :x_component_sigma_sq  => x_component_sigma_sq,
        :x_component           => x_component)

    var_approximations = Dict(
        x                     => qx,
        x_sigma_sq            => qx_sigma_sq,
        x_component_mu        => qx_component_mu,
        x_component_sigma_sq  => qx_component_sigma_sq,
        x_component           => qx_component)

    data = Dict(likapprox => Float32[])

    return vars, var_approximations, data
end


EXTRUDER_MODELS["expression"] = estimate_expression
EXTRUDER_MODELS["splicing"]   = estimate_splicing_log_ratio
