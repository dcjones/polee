
# This is a straightforward implementation of the technique from:
#   Li, Ping, Art Owen, and Cun-Hui Zhang. 2012. “One Permutation Hashing.” In
#   Advances in Neural Information Processing Systems 25, edited by F. Pereira,
#   C. J. C. Burges, L. Bottou, and K. Q. Weinberger, 3113–21. Curran Associates,
#   Inc.


"""
A h-partition sketch for k-mers.
"""
struct KmerSketch{H, K}
    minhash::Vector{UInt64}
end


"""
Construct an empty sketch.
"""
function KmerSketch{H,K}() where {H, K}
    return KmerSketch{H,K}(zeros(UInt64, H))
end


"""
Insert a k-mer into the sketch.
"""
function Base.push!(sketch::KmerSketch{H, K}, x::DNAMer{K}) where {H, K}
    h = hash(x)
    i = 1 + h % H

    if sketch.minhash[i] == 0 || sketch.minhash[i] > h + 1
        sketch.minhash[i] = h + 1
    end
end


function Base.copy(sketch::KmerSketch{H, K}) where {H, K}
    return KmerSketch{H,K}(copy(sketch.minhash))
end


function approximate_jaccard(a::KmerSketch{H, K}, b::KmerSketch{H, K}) where {H, K}
    mat = 0
    empt = 0
    for (u, v) in zip(a.minhash, b.minhash)
        if u == v
            if u == 0
                empt += 1
            else
                mat += 1
            end
        end
    end
    J =  Float32(mat) / Float32(H - empt)
    return J
end


function Base.union!(a::KmerSketch{H, K}, b::KmerSketch{H, K}) where {H, K}
    for i in 1:H
        if a.minhash[i] == 0 || a.minhash[i] > b.minhash[i]
            a.minhash[i] = b.minhash[i]
        end
    end
    return a
end


function Base.union(a::KmerSketch{H, K}, b::KmerSketch{H, K}) where {H, K}
    return union!(copy(a), b)
end


function Base.isless(a::KmerSketch, b::KmerSketch)
    for (u, v) in zip(a.minhash, b.minhash)
        if u < v
            return true
        elseif u > v
            return false
        end
    end
    return false # a == v
end


