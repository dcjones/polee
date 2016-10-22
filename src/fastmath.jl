
# Implement much faster approximate log functions

using SIMD

typealias FloatVec Vec{4, Float32}
typealias IntVec Vec{4, Int32}

typealias FloatTuple NTuple{4, VecElement{Float32}}
typealias IntTuple NTuple{4, VecElement{Int32}}

const ir_fv_type = "<4 x float>"
const ir_iv_type = "<4 x i32>"
const ir_i1v_type = "<4 x i1>"

function ir_set1(n, typ, val)
    out = IOBuffer()
    print(out, "< ")
    print(out, typ, " ", val)
    for i in 2:n
        print(out, ", ", typ, " ", val)
    end
    print(out, " >")
    s = takebuf_string(out)
    return s
end

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

function fastlog{N}(x::NTuple{N,VecElement{Float32}})
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
        FloatTuple, Tuple{FloatTuple}, x)
    return ans
end


function fastlog(x::FloatVec)
    exp = convert(FloatVec, reinterpret(IntVec, x) >> 23)

    addcst = ifelse(x > 0.0f0, FloatVec(-89.970756366f0), FloatVec(-Inf))

    x = reinterpret(FloatVec,
                (reinterpret(IntVec, x) & Int32(0x007fffff)) | Int32(0x3f800000))

    return x * (3.529304993f0 + x * (-2.461222105f0 +
          x * (1.130626167f0 + x * (-0.288739945f0 +
          x * 3.110401639f-2)))) + (addcst + 0.69314718055995f0*exp)
end


const ir_exp_c1 = "0x4167154760000000" # 12102203.1615614f0
const ir_exp_c2 = "0x41cfc00000000000" # 1065353216.0f0
const ir_exp_c3 = "0x41dfe00000000000" # 2139095040.f0

const ir_exp_c4 = "0x3fe0552ce0000000" # 0.510397365625862338668154f0
const ir_exp_c5 = "0x3fd3e20820000000" # 0.310670891004095530771135f0
const ir_exp_c6 = "0x3fc585b960000000" # 0.168143436463395944830000f0
const ir_exp_c7 = "0xbf6799c2a0000000" # -2.88093587581985443087955f-3
const ir_exp_c8 = "0x3f8bff8dc0000000" # 1.3671023382430374383648148f-2

function fastexp{N}(x::NTuple{N,VecElement{Float32}})
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
        FloatTuple, Tuple{FloatTuple}, x)

    return ans
end


function fastexp(x::FloatVec)
  val2 = 12102203.1615614f0*x + 1065353216.0f0;

  exp_cst1 = FloatVec(2139095040.f0)
  exp_cst2 = FloatVec(0.0f0)
  val3 = ifelse(val2 < exp_cst1, val2, exp_cst1)
  val4 = ifelse(val3 > exp_cst2, val3, exp_cst2)

  val4i = convert(IntVec, val4)
  xu = reinterpret(FloatVec, val4i & Int32(0x7f800000))
  b = reinterpret(FloatVec, (val4i & Int32(0x007fffff)) | Int32(0x3f800000))

  return xu * (0.510397365625862338668154f0 + b *
            (0.310670891004095530771135f0 + b *
             (0.168143436463395944830000f0 + b *
              (-2.88093587581985443087955f-3 + b *
               1.3671023382430374383648148f-2))))
end


function f(xs::Vector{Float32})
    xsv = reinterpret(FloatTuple, xs)
    @inbounds for i in length(xsv)
        xsv[i] = fastlog(xsv[i])
    end
end

function g(xs)
    @inbounds for i in 1:n
        xs[i] = log(xs[i])
    end
end

#n = 40000000
const n = 400000
const xs = rand(Float32, n)
#@show typeof(xs)

const x = vload(FloatVec, xs, 1)
@show fastexp(x)
@show fastexp(x.elts)

@show @code_native fastexp(x.elts)

#@show @code_llvm f(xs)


#ys = copy(xs)
#f(ys)
#ys = copy(xs)
#@time f(ys)

#ys = copy(xs)
#g(ys)
#ys = copy(xs)
#@time g(ys)



