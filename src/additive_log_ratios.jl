
# additive log-ratio transformation
struct ALRTransform
    # index of reference element
    refidx::Int
end

"""
Transfrom real numbers ys to simplex constrain vector xs using ALR
transformation.
"""
function alr_transform!(t::ALRTransform, ys::Vector, xs::Vector,
                        ::Type{Val{GRADONLY}}) where {GRADONLY}
    n = length(xs)
    for i in 1:n
        if i == t.refidx
            xs[i] = 1.0
        else
            j = i < t.refidx ? i : i - 1
            xs[i] = exp(ys[j])
        end
    end
    xs ./= sum(xs)

    ladj = 0.0
    if !GRADONLY
        ladj = sum(ys) - log(1 + sum(exp.(ys)))
    end

    return ladj
end


"""
Transfrom simplex constrained xs to unconstrained real numbers ys.
"""
function inverse_alr_transform!(t::ALRTransform, xs::Vector, ys::Vector,
                                ::Type{Val{GRADONLY}}) where {GRADONLY}
    n = length(xs)
    for i in 1:n
        if i == t.refidx
            continue
        end
        j = i < t.refidx ? i : i - 1
        ys[j] = log(xs[j] / xs[t.refidx])
    end
end


function alr_transform_gradients!(t::ALRTransform, ys::Vector, xs::Vector,
                                  y_grad::Vector, x_grad::Vector)

    n = length(xs)
    ys_exp_sum = 1 + sum(exp.(ys))

    c = 0.0
    for i in 1:n
        c += xs[i] * x_grad[i]
    end

    for i in 1:n
        if i == t.refidx
            continue
        end
        j = i < t.refidx ? i : i - 1

        y_grad[j] = xs[i] * x_grad[i] - xs[i] * c

        # ladj gradients
        y_grad[j] += 1 - xs[i]
    end
end

