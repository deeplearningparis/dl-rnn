def accumulate_polynomial(coef, cur_polynomial, cur_power, xx):
    return cur_polynomial + coef * cur_power, cur_power * xx

(r_poly_expr, _), updates = theano.reduce(accumulate_polynomial,
                                          sequences=coefs, 
                                          outputs_info=[np.float32(0.), np.float32(1.)],
                                          non_sequences=x)

f_r_poly = theano.function([x], r_poly_expr)