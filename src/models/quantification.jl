


function estimate_expression(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_expression(input)
    elseif input.feature == :gene
        return estimate_gene_expression(input)
    elseif input.feature == :splicing
        return estimate_splicing_expression(input)
    else
        error("Expression estimates for $(feature)s not supported")
    end
end


function estimate_transcript_expression(input::ModelInput)
    num_samples, n = input.x0[:get_shape]()[:as_list]()
    x, x_mu_param, x_sigma_param, x_mu, likapprox_musigma =
        transcript_quantification_model(input)

    println("Estimating...")

    qx_mu_param = tf.Variable(tf.log(input.x0))
    qx_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0)))
    qx = edmodels.MultivariateNormalDiag(qx_mu_param, qx_sigma_param)

    qx_mu_mu_param = tf.Variable(input.x0[1])
    qx_mu_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([n], -1.0f0)))
    qx_mu = edmodels.MultivariateNormalDiag(qx_mu_mu_param, qx_mu_sigma_param)

    inference = ed.KLqp(PyDict(Dict(x => qx, x_mu => qx_mu)),
                        data=PyDict(Dict(likapprox_musigma => input.likapprox_musigma)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    # optimizer = tf.train[:MomentumOptimizer](1e-7, 0.8)
    # inference[:run](n_iter=250, optimizer=optimizer)
    inference[:run](n_iter=500, optimizer=optimizer)

    sess = ed.get_session()
    qx_mu_value    = sess[:run](qx_mu_param)
    qx_sigma_value = sess[:run](qx_sigma_param)

    # est = sess[:run](tf.nn[:softmax](qx_mu_param, dim=-1))
    # est = sess[:run](tf.divide(input.x0, input.likapprox_efflen))
    est = sess[:run](input.x0)
    efflens = sess[:run](input.likapprox_efflen)
    @show extrema(est)
    @show extrema(efflens)

    # reset session and graph to free up memory
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    @show minimum(qx_mu_value), median(qx_mu_value), maximum(qx_mu_value)
    @show minimum(qx_sigma_value), median(qx_sigma_value), maximum(qx_sigma_value)

    # TODO: this should be a temporary measure until we decide exactly how
    # results should be reported. Probably in sqlite or something.

    write_estimates("estimates.csv", input.sample_names, est)

    open("efflen.csv", "w") do out
        println(out, "transcript_num,efflen")
        for (i, efflen) in enumerate(efflens)
            println(out, i, ",", efflen)
        end
    end


    exit()

    return qx_mu_value, qx_sigma_value
end


function estimate_gene_expression(input::ModelInput)
    m, I, J, names = gene_feature_matrix(input.ts, input.ts_metadata)
    F = tf.SparseTensor(indices=cat(2, I-1, J-1), values=tf.ones(length(I)),
                        dense_shape=[m, length(input.ts)])

    qy_mu_value, qy_sigma_value =
        estimate_feature_expression(input.likapprox_data, input.y0, input.sample_factors, F)
end


function estimate_splicing_expression(input::ModelInput)
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
        "insert into splicing_feature_including_transcripts values (?1, ?2)")
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

    # build feature matrix
    I = Int[]
    J = Int[]
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        @assert !isempty(intron.metadata)
        @assert !isempty(flanks.metadata[3])

        for id in flanks.metadata[3]
            push!(I, 2*i-1)
            push!(J, id)
        end

        for id in intron.metadata
            push!(I, 2*i)
            push!(J, id)
        end
    end
    m = 2*length(cassette_exons)

    F = tf.SparseTensor(indices=cat(2, I-1, J-1), values=tf.ones(length(I)),
                        dense_shape=[m, length(input.ts)])

    qy_mu_value, qy_sigma_value =
        estimate_feature_expression(input.likapprox_data, input.y0, input.sample_factors, F)
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


function transcript_quantification_model(input::ModelInput)
    num_samples, n = input.x0[:get_shape]()[:as_list]()

    # y_mu: pooled mean
    x_mu_mu0 = tf.constant(log(1/n), shape=[n])
    x_mu_sigma0 = tf.constant(10.0, shape=[n])
    x_mu = edmodels.MultivariateNormalDiag(x_mu_mu0, x_mu_sigma0)

    # x_sigma: variance around pooled mean
    x_sigma_mu0 = tf.constant(0.0, shape=[n])
    x_sigma_sigma0 = tf.constant(1.0, shape=[n])
    x_log_sigma = edmodels.MultivariateNormalDiag(x_sigma_mu0, x_sigma_sigma0)
    x_sigma = tf.exp(x_log_sigma)

    # y: quantification
    x_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                           tf.expand_dims(x_mu, 0))

    x_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(x_sigma, 0))
    # y_sigma_param = tf.Print(y_sigma_param, [tf.reduce_min(y_sigma_param), tf.reduce_max(y_sigma_param)], "Y_SIGMA SPAN")

    x = edmodels.MultivariateNormalDiag(x_mu_param, x_sigma_param)

    likapprox_musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                    x=x,
                    efflens=input.likapprox_efflen,
                    As=input.likapprox_As,
                    node_parent_idxs=input.likapprox_parent_idxs,
                    node_js=input.likapprox_js,
                    value=input.likapprox_musigma)

    return x, x_mu_param, x_sigma_param, x_mu, likapprox_musigma
end


function estimate_feature_expression(likapprox_data, y0, sample_factors, features)
    num_samples, n = y0[:get_shape]()[:as_list]()
    num_features = features[:get_shape]()[:as_list]()[1]
    y, y_mu_param, y_sigma_param, y_mu, likapprox =
        transcript_quantification_model(likapprox_data, y0)

    # TODO: I'm getting INFS. The issue is extremely large SIGMA values that
    # occur for some reason.
    #
    # Update: still getting NaNs, despite fixing the issue with extreme sigmas.
    # have no idea why that's happening.
    #

    # We effectively want to sum log-normal variates to arrive at feature
    # expression. We approximate that with another log-normal by matching
    # moments (this is sometimes called the Fenton-Wilkinson method)

    # y_sigma_param = tf.Print(y_sigma_param, [tf.reduce_min(y_sigma_param), tf.reduce_max(y_sigma_param)], "Y_SIGMA SPAN")
    y_var = tf.square(y_sigma_param)
    y_var = tf.Print(y_var, [tf.reduce_min(y_var), tf.reduce_max(y_var)], "Y_VAR SPAN")
    y_mu_param = tf.Print(y_mu_param, [tf.reduce_min(y_mu_param), tf.reduce_max(y_mu_param)], "Y_MU SPAN")

    tmp = tf.add(y_mu_param, tf.divide(y_var, 2))
    tmp = tf.Print(tmp, [tf.reduce_min(tmp), tf.reduce_max(tmp)], "TMP SPAN")
    y_mu_exp_adj1 = tf.exp(tmp)

    # y_mu_exp_adj1 = tf.exp(tf.add(y_mu_param, tf.divide(y_var, 2)))
    y_mu_exp_adj1 = tf.Print(y_mu_exp_adj1, [tf.reduce_min(y_mu_exp_adj1), tf.reduce_max(y_mu_exp_adj1)], "MU_EXP_ADJ1 SPAN")
    y_features_mu_exp_part1 =
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y_mu_exp_adj1, adjoint_b=true))

    y_mu_exp_adj2 = tf.multiply(tf.exp(tf.add(tf.multiply(2.0f0, y_mu_param), y_var)),
                                tf.subtract(tf.exp(y_var), 1.0f0))
    # y_mu_exp_adj2 = tf.Print(y_mu_exp_adj2, [tf.reduce_min(y_mu_exp_adj2), tf.reduce_max(y_mu_exp_adj2)], "MU_EXP_ADJ1 SPAN")
    y_features_mu_exp_part2 =
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y_mu_exp_adj2, adjoint_b=true))

    var_numer = y_features_mu_exp_part1
    var_denom = tf.square(y_features_mu_exp_part2)

    # var_numer = tf.Print(var_numer, [tf.reduce_min(var_numer), tf.reduce_max(var_numer)], "NUMER SPAN")
    # var_denom = tf.Print(var_denom, [tf.reduce_min(var_denom), tf.reduce_max(var_denom)], "DENOM SPAN")

    y_features_var_param = tf.log(tf.add(1.0f0, tf.divide(var_numer, var_denom)))
    # tmp = tf.divide(var_numer, var_denom)
    # tmp = tf.Print(tmp, [tf.reduce_min(tmp), tf.reduce_max(tmp)], "TMP SPAN")
    # y_features_var_param = tf.log(tf.add(1.0f0, tmp))

    y_features_sigma_param = tf.sqrt(y_features_var_param)

    y_features_mu_param = tf.subtract(tf.log(y_features_mu_exp_part1),
                                      tf.divide(tf.square(y_features_sigma_param), 2))

    y_features_mu_param = tf.Print(y_features_mu_param,
        [tf.reduce_min(y_features_mu_param), tf.reduce_max(y_features_mu_param)], "FEATURE MU SPAN")
    y_features_sigma_param = tf.Print(y_features_sigma_param,
        [tf.reduce_min(y_features_sigma_param), tf.reduce_max(y_features_sigma_param)], "FEATURE SIGMA SPAN")

    #=
    y_features_mu_param =
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y_mu_param, adjoint_b=true))
    y_features_sigma_param =
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y_sigma_param, adjoint_b=true))
    =#
    y_features = edmodels.MultivariateNormalDiag(y_features_mu_param, y_features_sigma_param)

    println("Estimating...")

    # y approximation
    qy_mu_param = tf.Variable(y0)
    qy_sigma_param = tf.identity(tf.Variable(tf.fill([num_samples, n], 0.1)))
    qy = edmodels.MultivariateNormalDiag(qy_mu_param, qy_sigma_param)

    # y_feature approximation
    qy_features_mu_param = tf.Variable(
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y0, adjoint_b=true)))
    qy_features_sigma_param =
        tf.nn[:softplus](tf.Variable(tf.fill([num_samples, num_features], -1.0f0)))
    qy_features = edmodels.MultivariateNormalDiag(qy_features_mu_param,
                                                  qy_features_sigma_param)
    # y_mu approximation
    qy_mu_mu_param = tf.Variable(y0[1])
    # qy_mu_sigma_param = tf.identity(tf.Variable(tf.fill([n], 0.1)))
    qy_mu_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([n], -1.0f0)))
    qy_mu = edmodels.MultivariateNormalDiag(qy_mu_mu_param, qy_mu_sigma_param)

    inference = ed.KLqp(PyDict(Dict(y => qy, y_features => qy_features, y_mu => qy_mu)),
                        data=PyDict(Dict(likapprox => likapprox_data)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    inference[:run](n_iter=10, optimizer=optimizer)

    sess = ed.get_session()
    qy_mu_value    = sess[:run](qy_features_mu_param)
    qy_sigma_value = sess[:run](qy_features_sigma_param)

    # reset session and graph to free up memory
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    return qy_mu_value, qy_sigma_value
end



EXTRUDER_MODELS["expression"] = estimate_expression
EXTRUDER_MODELS["splicing"]   = estimate_splicing_log_ratio

