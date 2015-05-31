import numpy as np
import theano
import theano.tensor as T
from theano import shared 

dtype = T.config.floatX

print("loading init.py")


def init_weight(shape, name, sample='uni', seed=None):
    rng = np.random.RandomState(seed)

    if sample == 'unishape':
        values = rng.uniform(
            low=-np.sqrt(6. / (shape[0] + shape[1])),
            high=np.sqrt(6. / (shape[0] + shape[1])),
            size=shape).astype(dtype)

    elif sample == 'svd':
        values = rng.uniform(low=-1., high=1., size=shape).astype(dtype)
        _, svs, _ = np.linalg.svd(values)
        # svs[0] is the largest singular value
        values = values / svs[0]

    elif sample == 'uni':
        values = rng.uniform(low=-0.1, high=0.1, size=shape).astype(dtype)
    
    elif sample == 'zero':
        values = np.zeros(shape=shape, dtype=dtype)

    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)

    return shared(values, name=name, borrow=True)

