def add_square_a_to_b(a, b):  # to be removed
    return a ** 2 + b

expr_square_sum, updates = theano.reduce(add_square_a_to_b,
                                         sequences=T.arange(n),
                                         outputs_info=np.cast['int32'](0))
f_square_sum = theano.function([n], expr_square_sum)
