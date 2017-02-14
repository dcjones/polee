
# This is a modification of the Base randn 

@inline function _randn(rng::AbstractRNG=Base.GLOBAL_RNG)
    @inbounds begin
        r = Base.Random.rand_ui52(rng)
        rabs = Int64(r>>1) # One bit for the sign
        idx = rabs & 0xFF
        x = ifelse(r % Bool, -rabs, rabs)*Base.Random.wi[idx+1]
        rabs < Base.Random.ki[idx+1] && return x # 99.3% of the time we return here 1st try
        return _randn_unlikely(rng, idx, rabs, x)::Float64
    end
end

# this unlikely branch is put in a separate function for better efficiency
function _randn_unlikely(rng, idx, rabs, x)
    @inbounds if idx == 0
        while true
            xx = -Base.Random.ziggurat_nor_inv_r*log(rand(rng))
            yy = -log(rand(rng))
            yy+yy > xx*xx && return (rabs >> 8) % Bool ?  -Base.Random.ziggurat_nor_r-xx : Base.Random.ziggurat_nor_r+xx
        end
    elseif (Base.Random.fi[idx] - Base.Random.fi[idx+1])*rand(rng) + Base.Random.fi[idx+1] < exp(-0.5*x*x)
        return x # return from the triangular area
    else
        return randn(rng)::Float64
    end
end
