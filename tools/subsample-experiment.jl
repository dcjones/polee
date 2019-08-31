#!/usr/bin/env julia

using YAML
using Random

# Things this has to be able to do:
#   * sample from a subset of tissues
#   * manually set the seed
#   * (approx) equal numbers from each tissue?


function print_usage()
    println(
        """
        Usage: ./subsample-experiment.jl experiment.yml factor seed train_count test_count
        For example:
            ./subsample-experiment.jl experiment.yml tissue 12345 100 20

        Outputs two files:
            training.yml and testing.yml
        """)
end


"""
Extremely simple and incomplete YAML printer.
"""
function print_yaml(output, item, level=0, indent_first=true)
    if isa(item, Array)
        for subitem in item
            print_indent(output, level)
            print(output, "- ")
            print_yaml(output, subitem, level+1, false)
        end
    elseif isa(item, Dict)
        for (i, (key, val)) in enumerate(item)
            if (i == 1 && indent_first) || i > 1
                print_indent(output, level)
            end
            print(output, key, ": ")
            if isa(val, Dict) || isa(val, Array)
                println(output)
                print_yaml(output, val, level+1, true)
            else
                print_yaml(output, val, level+1, false)
            end
        end
    else
        println(output, item)
    end
end


function print_indent(output, level)
    for _ in 1:level
        print(output, "  ")
    end
end


function main()
    if length(ARGS) < 5
        print_usage()
        exit(1)
    end

    experiment_filename = ARGS[1]
    factor = ARGS[2]
    Random.seed!(parse(Int, ARGS[3]))
    train_count = parse(Int, ARGS[4])
    test_count = parse(Int, ARGS[5])

    spec = YAML.load_file(experiment_filename)

    # index spec by factor
    samples_by_factor = Dict{String, Vector{Any}}()
    num_samples = 0
    for sample in spec["samples"]
        if haskey(sample, "factors") && haskey(sample["factors"], factor)
            sample_factor = sample["factors"][factor]
            values = get!(() -> Any[], samples_by_factor, sample_factor)
            push!(values, sample)
            num_samples += 1
        end
    end

    # sample subsets, preserving a much as possible the proportion of each tissue
    num_factors = length(samples_by_factor)

    train_prop = train_count / num_samples
    test_prop = test_count / num_samples

    if train_prop + test_prop > 1
        train_prop = train_prop / (train_prop + test_prop)
        test_prop = test_prop / (train_prop + test_prop)
    end

    factor_training_counts = Dict{String, Int}()
    factor_testing_counts = Dict{String, Int}()

    for (factor_value, samples) in samples_by_factor
        if length(samples) < 2
            error("Less that two samples in factor ", factor_value)
        end

        factor_training_counts[factor_value] =
            floor(Int, train_prop * length(samples))
        factor_testing_counts[factor_value] =
            floor(Int, test_prop * length(samples))
    end

    # allocate any remainders
    train_count_remainder = train_count - sum(values(factor_training_counts))
    test_count_remainder = test_count - sum(values(factor_testing_counts))

    while train_count_remainder > 0 || test_count_remainder > 0
        allocated = false
        for (factor_value, samples) in samples_by_factor
            allocated_count =
                factor_training_counts[factor_value] +
                factor_testing_counts[factor_value]

            if length(samples) > allocated_count && train_count_remainder > 0
                factor_training_counts[factor_value] += 1
                train_count_remainder -= 1
                allocated_count += 1
                allocated = true
            end

            if length(samples) > allocated_count && test_count_remainder > 0
                factor_testing_counts[factor_value] += 1
                test_count_remainder -= 1
                allocated_count += 1
                allocated = true
            end
        end

        if !allocated # nothing else to allocate
            break
        end
    end

    # make sure we didn't fuck up
    for (factor_value, samples) in samples_by_factor
        allocated_count =
            factor_training_counts[factor_value] +
            factor_testing_counts[factor_value]
        @assert length(samples) <= allocated_count
    end

    training_samples = Any[]
    testing_samples = Any[]
    for (factor_value, samples) in samples_by_factor
        idx = shuffle(1:length(samples))

        append!(
            training_samples,
            samples[idx[1:factor_training_counts[factor_value]]])

        append!(
            testing_samples,
            samples[idx[end-factor_testing_counts[factor_value]+1:end]])
    end

    spec["samples"] = training_samples
    open("training.yml", "w") do output
        print_yaml(output, spec)
    end

    spec["samples"] = testing_samples
    open("testing.yml", "w") do output
        print_yaml(output, spec)
    end
end


main()

