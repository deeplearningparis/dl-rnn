def accumulate_powers(i, cur_result, powers):
    return cur_result + i ** powers

acc_powers, updates = theano.reduce(accumulate_powers,
                                    sequences=T.arange(n),
                                    outputs_info=T.zeros_like(powvec, dtype='int32'),
                                    non_sequences=powvec)

f_acc_powers = theano.function([n, powvec], acc_powers)
