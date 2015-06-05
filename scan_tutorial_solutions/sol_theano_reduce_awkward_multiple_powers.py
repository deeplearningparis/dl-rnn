def accumulate_all_powers(i, *args):
    # half of args is current values, the other half is non_sequences
    current_values = args[:len(args) / 2]
    powers = args[len(args) / 2:]
    
    output = [i ** power + cur_val
                  for power, cur_val in zip(powers, current_values)]
    return output

all_expressions, updates = theano.reduce(accumulate_all_powers,
                                         sequences=T.arange(n),
                                         outputs_info=[np.int32(0)] * len(powers),
                                         non_sequences=powers)

f_all_powers = theano.function([n], all_expressions)
