def generate_monomial(coefficient, counter, xx):
    return coefficient * xx ** counter

monomial_expr, updates = theano.map(generate_monomial,
                                    sequences=[coefs, T.arange(coefs.shape[0])],
                                    non_sequences=x)

f_monomials = theano.function([x], monomial_expr)
