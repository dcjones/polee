
type RNASeqSample
    m::Int
    n::Int
    X::Union{MKLSparseMatrixCSC, SparseMatrixCSC, RSBMatrix}
end


function Base.size(s::RNASeqSample)
    return (s.m, s.n)
end


function Base.read(filename::String, ::Type{RNASeqSample})
    input = h5open(filename, "r")
    m = read(input["m"]) # number of reads
    n = read(input["n"]) # number of transcripts
    colptr = read(input["colptr"])
    rowval = read(input["rowval"])
    nzval = read(input["nzval"])
    X = RSBMatrix(SparseMatrixCSC(m, n, colptr, rowval, nzval))
    #mklX = MKLSparseMatrixCSC(X)

    return RNASeqSample(m, n, X)
end



