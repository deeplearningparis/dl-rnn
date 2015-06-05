def acc_ints_and_squares(i, cur_ints, cur_squares):
    return (i ** 2 + i) / 2 + 1, i + cur_ints, i ** 2 + cur_squares

scan_ints_and_squares, updates = theano.scan(acc_ints_and_squares,
                                             sequences=T.arange(n),
                                             outputs_info=[None, np.int32(0), np.int32(0)])

f_scan_ints_and_squares = theano.function([n], scan_ints_and_squares)
