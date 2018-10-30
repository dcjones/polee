#!/usr/bin/env julia

function read_knn_graph(filename)
    input = open(filename)
    readline(input) # header
    edges = Set{Tuple{String,String}}()
    for line in eachline(input)
        u, v = split(line, ',')
        push!(edges, (u, v))
    end
    close(input)
    return edges
end

function main()
    filename_a, filename_b = ARGS

    as = read_knn_graph(filename_a)
    bs = read_knn_graph(filename_b)

    @assert length(as) == length(bs)

    count = 0
    for a in as
        if in(a, bs)
            count += 1
        end
    end

    println(count/length(as))
end

main()
