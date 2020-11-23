
"""
Load necessary info for likelihood approximation from salmon output.
"""
function load_salmon_likelihood(salmon_dir::String, transcript_ids::Vector{String})

    # for mapping salmon transcript indexes to the same transcript indexes
    # used by the polee transformation
    tid_map = Dict{String, Int}()
    for (i, tid) in enumerate(transcript_ids)
        tid_map[tid] = i
    end

    # get transcript_ids and probabilities from aux_info/eq_classes.txt.gz
    eqc_filename = joinpath(salmon_dir, "aux_info", "eq_classes.txt.gz")
    salmon_transcript_ids = String[]

    if !isfile(eqc_filename)
        error("Missing likelihood data. Please run salmon quand with '-d'")
    end

    I = Int[]
    J = Int[]
    V = Float32[]
    ks = Int[]
    efflens = Float32[]

    input = open(eqc_filename)

    stream = GzipDecompressorStream(input)
    n = parse(Int, readline(stream))
    m = parse(Int, readline(stream))
    for i in 1:n
        push!(salmon_transcript_ids, readline(stream))
    end

    if Set(salmon_transcript_ids) != Set(transcript_ids)
        error(
            """
            'salmon index' and 'polee fit-tree' were used with different sets of transcripts.
            You may need to run 'salmon index' with '--keepDuplicates'.
            """)
    end

    # get effective lengths from quant.sf
    quant_filename = joinpath(salmon_dir, "quant.sf")
    resize!(efflens, n)
    open(quant_filename) do input
        readline(input) # header
        for line in eachline(input)
            row = split(line, '\t')
            efflens[tid_map[row[1]]] = parse(Float32, row[3])
        end
    end

    l = 1
    for i in 1:m
        row = split(readline(stream), '\t')
        nval = parse(Int, row[1])

        push!(ks, parse(Int, row[1+2*nval+1]))

        for j in 1:nval
            push!(I, i)
            push!(J, tid_map[salmon_transcript_ids[1+parse(Int, row[1+j])]])
            push!(V, parse(Float32, row[1+nval+j]))
        end

        if length(row) < 2 + 2*nval
            error("Missing likelihood data. Please run salmon quand with '-d'")
        end
    end

    close(input)

    X = sparse(I, J, V, m, n)
    return X, ks, efflens
end

