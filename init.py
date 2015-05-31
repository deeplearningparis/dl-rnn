import numpy as np
import theano
import theano.tensor as T
from theano import shared 

dtype=T.config.floatX

print "loaded init.py"

def init_weight(shape, name, sample='uni'):
    if sample=='unishape':
        return shared(value=np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (shape[0] + shape[1])),
                high=np.sqrt(6. / (shape[0] + shape[1])),
                size=shape), dtype=dtype), 
                    name=name, borrow=True)
    
    if sample=='svd':
        values = np.ndarray(shape, dtype=dtype)
        for dx in xrange(shape[0]):
            vals = np.random.uniform(low=-1., high=1.,  size=(shape[1],))
            values[dx,:] = vals
        _,svs,_ = np.linalg.svd(values)
        #svs[0] is the largest singular value                      
        values = values / svs[0]
        return shared(values, name=name, borrow=True)
    
    if sample=='uni':
        return shared(value=np.asarray(np.random.uniform(low=-0.1,high=0.1, size=shape), dtype=dtype), 
                      name=name, borrow=True)
    
    if sample=='zero':
        return shared(value=np.zeros(shape=shape, dtype=dtype), 
                      name=name, borrow=True)
    
    
    raise "error bad sample technique"
