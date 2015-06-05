def fib_acc(old, older):
    return old + older

fib_expr, updates = theano.scan(
    fib_acc,
    sequences=None,
    outputs_info=[dict(initial=np.int32([0, 1]), taps=[-1, -2])],
    n_steps=n)

f_fib_scan = theano.function([n], fib_expr)
