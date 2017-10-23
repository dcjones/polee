
# Sinh-arcsinh transformations based on
#     Jones, M. C., and Arthur Pewsey. "Sinh-arcsinh distributions."
#     Biometrika 96.4 (2009): 761-780.


"""
One parameter sinh/asinh transformation from zs0 -> zs
"""
function sinh_asinh_transform!{GRADONLY}(alpha, zs0, zs, ::Type{Val{GRADONLY}})
    n = length(alpha)+1
    ladj = 0.0f0
    for i in 1:n-1
        c = alpha[i] + asinh(zs0[i])
        zs[i] = sinh(c)

        if !GRADONLY
            ladj += cosh(c) - 0.5 * log(1 + zs0[i]^2)
        end
    end

    return ladj
end


"""
One parameter sinh/asinh transformation gradient wrt alpha.
"""
function sinh_asinh_transform!(zs0, zs, alpha, z_grad, alpha_grad)
    n = length(alpha)+1
    for i in 1:n-1
        dz_dalpha = cosh(alpha[i] + asinh(zs0[i]))
        alpha_grad[i] += dz_dalpha * z_grad[i]

        # ladj gradient
        alpha_grad[i] += zs[i] / dz_dalpha
    end
end

