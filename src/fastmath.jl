
# Implement much faster approximate log functions

module FastMath

export FloatVec, IntVec, logistic, fillpadded

typealias FloatVec NTuple{8, VecElement{Float32}}
typealias IntVec NTuple{8, VecElement{Int32}}

const ir_fv_type = "<8 x float>"
const ir_iv_type = "<8 x i32>"
const ir_i1v_type = "<8 x i1>"


function ir_set1(n, typ, val)
    out = IOBuffer()
    print(out, "< ")
    print(out, typ, " ", val)
    for i in 2:n
        print(out, ", ", typ, " ", val)
    end
    print(out, " >")
    return takebuf_string(out)
end


# generate a constant vector with incrementing entries
function ir_count(n, typ, from=1)
    out = IOBuffer()
    print(out, "< ")
    print(out, typ, " ")
        @printf(out, "%0.f", from)
    for i in 1:n-1
        print(out, ", ", typ, " ")
        @printf(out, "%0.f", from + i)
    end
    print(out, " >")
    return takebuf_string(out)
end


@inline function Base.zero{N}(::Type{NTuple{N,VecElement{Float32}}})
    return fill(NTuple{N,VecElement{Float32}}, 0.0f0)
end


"""
Allocate a Float32 vector padded to have length divisible by N, with any extra
entries set to `pad`.
"""
function fillpadded{N}(::Type{NTuple{N,VecElement{Float32}}}, v, n, pad=0.0f0)
    m = (div(n - 1, N) + 1) * N # ceil(n / N)
    xs = fill(Float32(v), m)
    for i in n+1:m
        xs[i] = pad
    end
    return xs
end


@inline function Base.:-{N}(x::NTuple{N,VecElement{Float32}})
    return Base.llvmcall(
        """
        %res = fsub $(ir_fv_type) $(ir_set1(N, "float", 0.0)), %0
        ret $(ir_fv_type) %res
        """,
        FloatVec, Tuple{FloatVec}, x)
end


@inline function Base.inv{N}(x::NTuple{N,VecElement{Float32}})
    return Base.llvmcall(
        """
        %res = fdiv $(ir_fv_type) $(ir_set1(N, "float", 1.0)), %0
        ret $(ir_fv_type) %res
        """,
        FloatVec, Tuple{FloatVec}, x)
end


@inline function Base.:+{N}(x::NTuple{N,VecElement{Float32}},
                            y::NTuple{N,VecElement{Float32}})
    return Base.llvmcall(
        """
        %res = fadd $(ir_fv_type) %0, %1
        ret $(ir_fv_type) %res
        """,
        FloatVec, Tuple{FloatVec, FloatVec}, x, y)
end


@inline function Base.:-{N}(x::NTuple{N,VecElement{Float32}},
                            y::NTuple{N,VecElement{Float32}})
    return Base.llvmcall(
        """
        %res = fsub $(ir_fv_type) %0, %1
        ret $(ir_fv_type) %res
        """,
        FloatVec, Tuple{FloatVec, FloatVec}, x, y)
end


@inline function Base.:./{N}(x::NTuple{N,VecElement{Float32}},
                             y::NTuple{N,VecElement{Float32}})
    return Base.llvmcall(
        """
        %res = fdiv $(ir_fv_type) %0, %1
        ret $(ir_fv_type) %res
        """,
        FloatVec, Tuple{FloatVec, FloatVec}, x, y)
end


@inline function Base.:.*{N}(x::NTuple{N,VecElement{Float32}},
                             y::NTuple{N,VecElement{Float32}})
    return Base.llvmcall(
        """
        %res = fmul $(ir_fv_type) %0, %1
        ret $(ir_fv_type) %res
        """,
        FloatVec, Tuple{FloatVec, FloatVec}, x, y)
end


@inline function Base.count{N}(::Type{NTuple{N,VecElement{Float32}}})
    return Base.llvmcall(
        """
        ret $(ir_fv_type) $(ir_count(N, "float"))
        """,
        FloatVec, Tuple{})
end


@inline function Base.fill{N}(::Type{NTuple{N,VecElement{Float32}}}, x::Float32)
    return Base.llvmcall(
        """
        %y1 = insertelement $(ir_fv_type) undef, float %0, i32 0
        %y2 = shufflevector $(ir_fv_type) %y1, $(ir_fv_type) undef,
            $(ir_iv_type) $(ir_set1(N, "i32", 0))
        ret $(ir_fv_type) %y2
        """,
        FloatVec, Tuple{Float32}, x)
end


# untested
@inline function Base.cumsum(x::NTuple{4,VecElement{Float32}})
    return Base.llvmcall(
        """
        %shuf1 = shufflevector
            $(ir_fv_type) %0,
            $(ir_fv_type) $(ir_set1(4, "float", 0.0)),
            $(ir_iv_type) < i32 4, i32 0, i32 1, i32 2 >
        %sum1 = fadd $(ir_fv_type) %0, %shuf1
        %shuf2 = shufflevector
            $(ir_fv_type) %sum1,
            $(ir_fv_type) $(ir_set1(4, "float", 0.0)),
            $(ir_iv_type) < i32 4, i32 4, i32 0, i32 1 >
        %sum2 = fadd $(ir_fv_type) %sum1, %shuf2
        ret $(ir_fv_type) %sum2
        """,
        FloatVec, Tuple{FloatVec}, x)
end


@inline function Base.cumsum(x::NTuple{8,VecElement{Float32}})
    return Base.llvmcall(
        """
        %shuf1 = shufflevector
            $(ir_fv_type) %0,
            $(ir_fv_type) < float 0.0, float undef, float undef, float undef,
                            float undef, float undef, float undef, float undef >,
            $(ir_iv_type) < i32 8, i32 0, i32 1, i32 2,
                            i32 3, i32 4, i32 5, i32 6 >
        %sum1 = fadd $(ir_fv_type) %0, %shuf1

        %shuf2 = shufflevector
            $(ir_fv_type) %sum1,
            $(ir_fv_type) < float 0.0, float undef, float undef, float undef,
                            float undef, float undef, float undef, float undef >,
            $(ir_iv_type) < i32 8, i32 8, i32 0, i32 1,
                            i32 2, i32 3, i32 4, i32 5 >
        %sum2 = fadd $(ir_fv_type) %sum1, %shuf2

        %shuf3 = shufflevector
            $(ir_fv_type) %sum2,
            $(ir_fv_type) < float 0.0, float undef, float undef, float undef,
                            float undef, float undef, float undef, float undef >,
            $(ir_iv_type) < i32 8, i32 8, i32 8, i32 8,
                            i32 0, i32 1, i32 2, i32 3 >
        %sum3 = fadd $(ir_fv_type) %sum2, %shuf3

        ret $(ir_fv_type) %sum3
        """,
        FloatVec, Tuple{FloatVec}, x)
end


@inline function Base.sum{N}(x::NTuple{N,VecElement{Float32}})
    return cumsum(x)[N].value
end

# Most of this was adapted from:
# http://gallium.inria.fr/blog/fast-vectorizable-math-approx/

# various float32 constants in a form that llvm ir understands
const ir_neginf = "0xfff0000000000000"
const ir_mask1  = "8388607" # 0x007fffff
const ir_mask2  = "1065353216" # 0x3f800000
const ir_mask3  = "2139095040" # 0x7f800000
const ir_log_a  = "0xc0567e20e0000000" # -89.970756366f0

const ir_log_c1 = "0x400c3c0440000000" # 3.529304993f0
const ir_log_c2 = "0xc003b09540000000" # -2.461222105f0
const ir_log_c3 = "0x3ff2170b80000000" # 1.130626167f0
const ir_log_c4 = "0xbfd27ab720000000" # -0.288739945f0
const ir_log_c5 = "0x3f9fd9bb40000000" # 3.110401639f-2
const ir_log_c6 = "0x3fe62e4300000000" # 0.69314718055995f0

function Base.log{N}(x::NTuple{N,VecElement{Float32}})
    ans = Base.llvmcall((
        """
        declare $(ir_fv_type) @llvm.fmuladd.f32($(ir_fv_type) %a, $(ir_fv_type) %b, $(ir_fv_type) %c)
        """,
        """
        %i0 = bitcast $(ir_fv_type) %0 to $(ir_iv_type)
        %expi = lshr $(ir_iv_type) %i0, $(ir_set1(N, "i32", 23))
        %exp = sitofp $(ir_iv_type) %expi to $(ir_fv_type)

        %lt0 = fcmp olt $(ir_fv_type) %0, $(ir_set1(N, "float", "0.0"))
        %addcst = select $(ir_i1v_type) %lt0,
            $(ir_fv_type) $(ir_set1(N, "float", ir_neginf)),
            $(ir_fv_type) $(ir_set1(N, "float", ir_log_a))

        %x1 = and $(ir_iv_type) %i0, $(ir_set1(N, "i32", ir_mask1))
        %x2 = or $(ir_iv_type) %x1, $(ir_set1(N, "i32", ir_mask2))
        %xf = bitcast $(ir_iv_type) %x2 to $(ir_fv_type)

        %y1 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %xf,
                $(ir_fv_type) $(ir_set1(N, "float", ir_log_c5)),
                $(ir_fv_type) $(ir_set1(N, "float", ir_log_c4)))

        %y2 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %xf,
                $(ir_fv_type) %y1,
                $(ir_fv_type) $(ir_set1(N, "float", ir_log_c3)))

        %y3 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %xf,
                $(ir_fv_type) %y2,
                $(ir_fv_type) $(ir_set1(N, "float", ir_log_c2)))

        %y4 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %xf,
                $(ir_fv_type) %y3,
                $(ir_fv_type) $(ir_set1(N, "float", ir_log_c1)))

        %z = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %exp,
                $(ir_fv_type) $(ir_set1(N, "float", ir_log_c6)),
                $(ir_fv_type) %addcst)

        %y5 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %xf,
                $(ir_fv_type) %y4,
                $(ir_fv_type) %z)

        ret $(ir_fv_type) %y5
        """),
        FloatVec, Tuple{FloatVec}, x)
    return ans
end


const ir_exp_c1 = "0x4167154760000000" # 12102203.1615614f0
const ir_exp_c2 = "0x41cfc00000000000" # 1065353216.0f0
const ir_exp_c3 = "0x41dfe00000000000" # 2139095040.f0

const ir_exp_c4 = "0x3fe0552ce0000000" # 0.510397365625862338668154f0
const ir_exp_c5 = "0x3fd3e20820000000" # 0.310670891004095530771135f0
const ir_exp_c6 = "0x3fc585b960000000" # 0.168143436463395944830000f0
const ir_exp_c7 = "0xbf6799c2a0000000" # -2.88093587581985443087955f-3
const ir_exp_c8 = "0x3f8bff8dc0000000" # 1.3671023382430374383648148f-2

function Base.exp{N}(x::NTuple{N,VecElement{Float32}})
    ans = Base.llvmcall((
        """
        declare $(ir_fv_type) @llvm.fmuladd.f32($(ir_fv_type) %a, $(ir_fv_type) %b, $(ir_fv_type) %c)
        """,
        """
        %v2 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %0,
                $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c1)),
                $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c2)))

        %ltc3 = fcmp olt $(ir_fv_type) %v2, $(ir_set1(N, "float", ir_exp_c3))
        %v3 = select $(ir_i1v_type) %ltc3,
               $(ir_fv_type) %v2,
               $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c3))

        %gt0 = fcmp ogt $(ir_fv_type) %v3, $(ir_set1(N, "float", 0.0))
        %v4 = select $(ir_i1v_type) %gt0,
               $(ir_fv_type) %v3,
               $(ir_fv_type) $(ir_set1(N, "float", 0.0))

        %v4i = fptosi $(ir_fv_type) %v4 to $(ir_iv_type)
        %xui = and $(ir_iv_type) %v4i, $(ir_set1(N, "i32", ir_mask3))
        %xu = bitcast $(ir_iv_type) %xui to $(ir_fv_type)

        %b1i = and $(ir_iv_type) %v4i, $(ir_set1(N, "i32", ir_mask1))
        %b2i = or $(ir_iv_type) %b1i, $(ir_set1(N, "i32", ir_mask2))
        %b = bitcast $(ir_iv_type) %b2i to $(ir_fv_type)

        %y1 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %b,
                $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c8)),
                $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c7)))

        %y2 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %b,
                $(ir_fv_type) %y1,
                $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c6)))

        %y3 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %b,
                $(ir_fv_type) %y2,
                $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c5)))

        %y4 = call $(ir_fv_type) @llvm.fmuladd.f32(
                $(ir_fv_type) %b,
                $(ir_fv_type) %y3,
                $(ir_fv_type) $(ir_set1(N, "float", ir_exp_c4)))

        %res = fmul $(ir_fv_type) %xu, %y4

        ret $(ir_fv_type) %res
        """),
        FloatVec, Tuple{FloatVec}, x)

    return ans
end


function logistic{N}(x::NTuple{N,VecElement{Float32}})
    return inv(fill(FloatVec, 1.0f0) + exp(-x))
end


if false
    function f(xs::Vector{Float32})
        xsv = reinterpret(FloatVec, xs)
        for i in 1:length(xsv)
            xsv[i] = fastlog(xsv[i])
        end
    end

    function g(xs)
        @inbounds for i in 1:n
            xs[i] = log(xs[i])
        end
    end

    #n = 40000000
    const n = 800000
    const xs = rand(Float32, n)
    #@show typeof(xs)

    #const x = vload(FloatVec, xs, 1)
    #@show fastexp(x)
    #@show fastexp(x.elts)
    #@show @code_native fastexp(x.elts)

    @show @code_native f(xs)


    #@show xs[1:10]
    ys = copy(xs)
    f(ys)
    ys = copy(xs)
    @time f(ys)
    #@show ys[1:10]

    ys = copy(xs)
    g(ys)
    ys = copy(xs)
    @time g(ys)
    #@show ys[1:10]
end

#const _x = VecElement{Float32}(1.2345)
#const _xv = (_x, _x, _x, _x, _x, _x, _x, _x)

#@show 1 / (1 + Base.exp(-_x.value))
#@show logistic(_xv)
#@show @code_native logistic(_xv)

#x = fill(FloatVec, 1.2345f0)
#@show logistic(x)
#@show @code_native(logistic(x))
#@show @code_native fill(FloatVec, 1.2345f0)

#x = count(FloatVec)
#@show x + x
#@show cumsum(x)
#@show cumsum([1, 2, 3, 4, 5, 6, 7, 8])
#@show @code_native cumsum(x)


end # module FastMath

