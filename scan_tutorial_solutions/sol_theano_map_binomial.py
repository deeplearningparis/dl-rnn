def generate_double_monomial(coefficient, counter, xx, yy, n):
    return coefficient * xx ** counter * yy ** (n - counter - 1)

double_monomial_expr, updates = theano.map(generate_double_monomial,
                                           sequences=[coefs, T.arange(coefs.shape[0])],
                                           non_sequences=[x, y, coefs.shape[0]])

f_binomial = theano.function([x, y], double_monomial_expr)