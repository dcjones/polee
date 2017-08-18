
function estimate_pca(input::ModelInput)
    if input.feature != :transcript
        error("PCA only implemented with transcripts")
    end

    num_samples, n = input.x0[:get_shape]()[:as_list]()
    num_components = 2

    w = edmodels.Normal(loc=tf.zeros([n, num_components]),
                        scale=tf.fill([n, num_components], 2.0f0))
    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.ones([num_samples, num_components]))
    # x = tf.transpose(tf.matmul(w, z, transpose_b=true))
    x = tf.matmul(z, w, transpose_b=true)
    @show x[:get_shape]()

    likapprox_musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                    x=x,
                    efflens=input.likapprox_efflen,
                    As=input.likapprox_As,
                    node_parent_idxs=input.likapprox_parent_idxs,
                    node_js=input.likapprox_js,
                    value=input.likapprox_musigma)

    qw = edmodels.Normal(loc=tf.Variable(tf.zeros([n, num_components])),
                         scale=tf.nn[:softplus](tf.Variable(tf.zeros([n, num_components]))))
    qz_loc = tf.Variable(tf.zeros([num_samples, num_components]))
    qz = edmodels.Normal(loc=qz_loc,
                         scale=tf.nn[:softplus](tf.Variable(tf.zeros([num_samples, num_components]))))

    # qz = edmodels.Normal(loc=tf.Variable(tf.random_normal([num_samples, num_components])),
    #                      scale=tf.nn[:softplus](tf.Variable(tf.random_normal([num_samples, num_components]))))

    inference = ed.KLqp(Dict(w => qw, z => qz),
                        data=Dict(likapprox_musigma => input.likapprox_musigma))

    optimizer = tf.train[:AdamOptimizer](1e-1)
    inference[:run](n_iter=1000, optimizer=optimizer)

    sess = ed.get_session()
    qz_loc_values = @show sess[:run](qz_loc)

    open("estimates.csv", "w") do out
        print(out, "sample,")
        println(out, join([string("pc", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.sample_names[i], '"', ',')
            println(out, join(qz_loc_values[i,:], ','))
        end
    end
end


# TODO: I want to compare against PCA on point estimates...

EXTRUDER_MODELS["pca"] = estimate_pca