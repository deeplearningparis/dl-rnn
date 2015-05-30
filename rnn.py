import numpy as np
import theano
import theano.tensor as T
from theano import shared 
from collections import OrderedDict

dtype=T.config.floatX

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

class RnnMiniBatch:
    def __init__(self, n_in, n_hid, n_out, lr=0.05, batch_size=64):   
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.W_in = init_weight((self.n_in, self.n_hid),'W_in')
        self.W_out = init_weight((self.n_hid, self.n_out),'W_out')
        self.W_rec = init_weight((self.n_hid, self.n_hid),'W_rec', 'svd')
                
        self.params = [self.W_in,self.W_out,self.W_rec]
        
        def step(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W_rec))
            y_t = T.nnet.softmax(T.dot(h_t, self.W_out))
            return [h_t, y_t]


        X = T.tensor3() # batch of sequence of vector
        Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null) 
        h0 = shared(np.zeros(shape=(batch_size,self.n_hid), dtype=dtype)) # initial hidden state         
        mask = 1. - X.sum(axis = 1)
        lr = shared(np.cast[dtype](lr))
        
        [h_vals, y_vals], _ = theano.scan(fn=step,        
                                          sequences=X,
                                          outputs_info=[h0, None])
        
        cost = (T.nnet.binary_crossentropy(y_vals.dimshuffle(1,0,2), Y) * mask).sum()        
        gparams = T.grad(cost, self.params)
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * lr
        
        self.train = theano.function(inputs = [X, Y], outputs = cost, updates=updates)
        self.predictions = theano.function(inputs = [X], outputs = y_vals[-1,0,:])
    
