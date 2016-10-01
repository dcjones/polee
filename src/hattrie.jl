
const libhattrie = Symbol("/usr/local/lib/libhat-trie.so")

type HatTrie <: Associative{String, UInt}
    ptr::Ptr{Void}

    function HatTrie()
        ht = new(ccall((:hattrie_create, libhattrie), Ptr{Void}, ()))
        finalizer(ht, _free)
        return ht
    end
end

function _free(ht::HatTrie)
    ccall((:hattrie_free, libhattrie), Void, (Ptr{Void},), ht.ptr)
end

function Base.get!(ht::HatTrie, key::String, default_::Integer)
    valptr = ccall((:hattrie_get, libhattrie), Ptr{UInt},
                   (Ptr{Void}, Ptr{UInt8}, Csize_t), ht.ptr, key, length(key))
    default = UInt(default_)
    val = unsafe_load(valptr)
    if val == 0
        unsafe_store!(valptr, default)
        return default
    else
        return val
    end
end

function Base.getindex(ht::HatTrie, key::String)
    # TODO
end

function Base.length(ht::HatTrie)
    return Int(ccall((:hattrie_size, libhattrie), Csize_t, (Ptr{Void},), ht.ptr))
end



