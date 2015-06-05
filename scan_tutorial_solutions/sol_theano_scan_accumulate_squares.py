scan_squares_expr, updates = theano.scan(add_square_a_to_b,
                                         sequences=T.arange(n),
                                         outputs_info=np.int32(0))
f_scan_squares = theano.function([n], scan_squares_expr)
