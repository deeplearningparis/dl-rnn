both_expressions, updates = theano.reduce(return_args_and_accumulate_squares_and_ints,
                                          sequences=T.arange(n), 
                                          outputs_info=[
            None, None, None, np.int32(0), np.cast['int32'](0)])
f_both = theano.function([n], both_expressions)
