
function counts_to_feature_log_props!(xs, efflens, pseudocount, features)
    xs ./= efflens
    xs ./= sum(xs)
    xs = reshape(xs, (1,length(xs)))
    xs *= features
    xs .+= pseudocount / 1f6
    map!(log, xs, xs)
    return xs
end


function read_kallisto_estimates(spec, pseudocount, features)
    filenames = [entry["kallisto"] for entry in spec["samples"]]
    xss = Array{Float32, 2}[]
    for filename in filenames
        h5open(filename) do input
            efflens = read(input["aux"]["eff_lengths"])
            xs = Vector{Float32}(read(input["est_counts"]))
            xs = counts_to_feature_log_props!(xs, efflens, pseudocount, features)
            push!(xss, xs)
        end
    end

    return vcat(xss...)
end


function read_kallisto_bootstrap_samples(spec, pseudocount, features)
    filenames = [entry["kallisto"] for entry in spec["samples"]]
    bootstraps = Array{Float32, 2}[]

    for filename in filenames
        xss = Array{Float32, 2}[]
        h5open(filename) do input
            efflens = read(input["aux"]["eff_lengths"])
            for dataset in input["bootstrap"]
                xs = Vector{Float32}(read(dataset))
                xs = counts_to_feature_log_props!(xs, efflens, pseudocount, features)
                push!(xss, xs)
            end
        end
        push!(bootstraps, vcat(xss...))
    end

    return bootstraps
end

