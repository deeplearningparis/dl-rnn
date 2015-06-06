a_vector = T.fvector()

def controlled_rotation(a, old):
    return T.exp(a) * rot.dot(old)

controlled_oscillation, updates = theano.scan(controlled_rotation,
                                             sequences=a_vector,
                                            outputs_info=np.float32([1., 0.]))

f_controlled_oscillation = theano.function([a_vector], controlled_oscillation)
