
function estimate_simple_mode(experiment_spec_filename, output_filename)
    experiment_spec = YAML.load_file(experiment_spec_filename)
    names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    num_samples = length(filenames)
    println("Read model specification with ", num_samples, " samples")

    n, musigma_data, y0 = load_samples(filenames)

    sess = ed.get_session()

    est = sess[:run](tf.multiply(1e6, tf.nn[:softmax](y0, dim=-1)))
    write_estimates(output_filename, names, est)
end

EXTRUDER_MODELS["simple-mode"] = estimate_simple_mode

