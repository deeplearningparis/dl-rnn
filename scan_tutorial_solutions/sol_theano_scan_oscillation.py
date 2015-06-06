def rotation_increment(old):
    return rot.dot(old)

oscillation, updates = theano.scan(
    rotation_increment,
    sequences=None,
    outputs_info=T.constant(np.float32([1., 0.])),
    n_steps=n)

f_oscillation = theano.function([n], oscillation)