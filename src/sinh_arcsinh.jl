
# Sinh-arcsinh transformations based on
#     Jones, M. C., and Arthur Pewsey. "Sinh-arcsinh distributions."
#     Biometrika 96.4 (2009): 761-780.


"""
One parameter sinh/asinh transformation from zs0 -> zs
"""
function sinh_asinh_transform!(alpha, zs0, zs, ::Val{compute_ladj}=Val(false)) where {compute_ladj}
    n = length(alpha)+1
    ladj = 0.0f0
    Threads.@threads for i in 1:n-1
        c = alpha[i] + asinh(zs0[i])
        zs[i] = sinh(c)

        if compute_ladj
            ladj += log(cosh(c)) - 0.5 * log1p(zs0[i]^2)
        end
    end

    return ladj
end


"""
One parameter sinh/asinh transformation gradient wrt alpha.
"""
function sinh_asinh_transform_gradients!(zs0, alpha, z_grad, alpha_grad)
    n = length(alpha)+1
    Threads.@threads for i in 1:n-1
        dz_dalpha = cosh(alpha[i] + asinh(zs0[i]))
        alpha_grad[i] += dz_dalpha * z_grad[i]

        # ladj gradient
        alpha_grad[i] += tanh(alpha[i] + asinh(zs0[i]))
    end
end
