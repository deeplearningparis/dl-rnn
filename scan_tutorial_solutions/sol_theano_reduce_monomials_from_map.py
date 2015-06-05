test, updates = theano.reduce(add,
                              sequences=monomial_expr,
                              outputs_info=np.float64(0))

f_poly = theano.function([x], test)
