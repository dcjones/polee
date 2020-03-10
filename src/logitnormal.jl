
logistic(x) = inv(1 + exp(-x))

logit(x) = log(x / (1 - x))



function logit_normal_transform!(mu, sigma, zs, ys, ::Val{compute_ladj}=Val(false)) where {compute_ladj}
    n = length(mu)+1
    ladj = 0.0f0
    for i in 1:n-1
        ys[i] = logistic(mu[i] + zs[i] * sigma[i])

        if compute_ladj
            ladj += log(sigma[i] * ys[i] * (1 - ys[i]))
        end
    end

    return ladj
end


function logit_normal_transform_gradients!(zs, ys, mu, sigma, y_grad, mu_grad, sigma_grad)
    n = length(mu)+1
    for i in 1:n-1
        dy_dmu = ys[i] * (1 - ys[i])
        mu_grad[i] += dy_dmu * y_grad[i]
        dy_dsigma = ys[i] * (1 - ys[i]) * zs[i]
        sigma_grad[i] += dy_dsigma * y_grad[i]

        # ladj gradients
        mu_grad[i] += 1 - 2*ys[i]
        sigma_grad[i] += 1/sigma[i] + zs[i] * (1 - 2*ys[i])
    end
end


function logit_normal_transform_gradients!(zs, ys, mu, sigma, y_grad, z_grad, mu_grad, sigma_grad)
    n = length(mu)+1
    for i in 1:n-1
        dy_dmu = ys[i] * (1 - ys[i])
        mu_grad[i] += dy_dmu * y_grad[i]

        dy_dsigma = ys[i] * (1 - ys[i]) * zs[i]
        sigma_grad[i] += dy_dsigma * y_grad[i]

        dy_dz = ys[i] * (1 - ys[i]) * sigma[i]
        z_grad[i] += dy_dz * y_grad[i]

        # ladj gradients
        mu_grad[i] += 1 - 2*ys[i]
        sigma_grad[i] += 1/sigma[i] + zs[i] * (1 - 2*ys[i])
        z_grad[i] += sigma[i] * (1 - 2*ys[i])
    end
end