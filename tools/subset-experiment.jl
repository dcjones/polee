#!/usr/bin/env julia

using YAML

function print_usage()
    println(
        """
        Usage: ./subset-experiment.jl experiment.yml factor values...
        For example:
            ./subset-experiment.jl experiment.yml tissue brain lung > subset.yml
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
    if length(ARGS) < 3
        print_usage()
        exit(1)
    end

    experiment_filename = ARGS[1]
    factor = ARGS[2]
    values = Set(ARGS[3:end])

    spec = YAML.load_file(experiment_filename)

    samples_subset = Any[]
    for sample in spec["samples"]
        if haskey(sample, "factors") && haskey(sample["factors"], factor) &&
                sample["factors"][factor] ∈ values
            push!(samples_subset, sample)
        end
    end

    spec["samples"] = samples_subset

    print_yaml(stdout, spec)
end

main()

