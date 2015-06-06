import numpy as np
import theano
import theano.tensor as T
from theano import shared 
from collections import OrderedDict
from init import init_weight

dtype=T.config.floatX

print "loaded rnn.py"

# Simple RNN class
# optional parameters: 
#  - activation: lambda x: x ; T.nnet.softmax ;T.nnet.sigmoid
#  - cost function: 'mse' 'bce' 'cce'

class Rnn:
    def __init__(self, n_in, n_hid, n_out, lr=0.05, single_output=True, output_activation=T.nnet.softmax, cost_function='nll'):   
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.W_in = init_weight((self.n_in, self.n_hid),'W_in')
        self.W_out = init_weight((self.n_hid, self.n_out),'W_out')
        self.W_rec = init_weight((self.n_hid, self.n_hid),'W_rec', 'svd')
        self.b_hid = shared(np.zeros(shape = n_hid, dtype=dtype))
        self.b_out = shared(np.zeros(shape = n_out, dtype=dtype))
        
        self.params = [self.W_in,self.W_out,self.W_rec,self.b_out,self.b_hid]
        
        self.activation = output_activation

        def step(x_t, h_tm1):
            h_t = T.tanh(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W_rec) + self.b_hid)
            y_t = self.activation(T.dot(h_t, self.W_out) + self.b_out)
            return [h_t, y_t]

        X = T.matrix() # sequence of vector
        Y = T.matrix() # sequence of vector 
        if single_output:
            Y = T.vector() 

        h0 = shared(np.zeros(shape=self.n_hid, dtype=dtype)) # initial hidden state
        lr = shared(np.cast[dtype](lr))
        
        [h_vals, y_vals], _ = theano.scan(fn=step,        
                                          sequences=X,
                                          outputs_info=[h0, None])

        if single_output:
            self.output = y_vals[-1]            
        else:
            self.output = y_vals
        
        cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
        nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))     
        mse = T.mean((self.output - Y) ** 2)

        cost = 0
        if cost_function == 'mse':
            cost = mse
        elif cost_function == 'cxe':
            cost = cxe
        else:
            cost = nll
            

        gparams = T.grad(cost, self.params)
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * lr
        
        self.loss = theano.function(inputs = [X, Y], outputs = cost)
        self.train = theano.function(inputs = [X, Y], outputs = cost, updates=updates)
        self.predictions = theano.function(inputs = [X], outputs = self.output)
        self.debug = theano.function(inputs = [X, Y], outputs = [X.shape, Y.shape, y_vals.shape, self.output.shape])


# Same class with MiniBatch support
class RnnMiniBatch:
    def __init__(self, n_in, n_hid, n_out, lr=0.05, batch_size=64, single_output=True, output_activation=T.nnet.softmax, cost_function='nll'):   
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.W_in = init_weight((self.n_in, self.n_hid),'W_in')
        self.W_out = init_weight((self.n_hid, self.n_out),'W_out')
        self.W_rec = init_weight((self.n_hid, self.n_hid),'W_rec', 'svd')
        self.b_hid = shared(np.zeros(shape = n_hid, dtype=dtype))
        self.b_out = shared(np.zeros(shape = n_out, dtype=dtype))

        self.params = [self.W_in,self.W_out,self.W_rec,self.b_out,self.b_hid]

        self.activation = output_activation

        def step(x_t, h_tm1):
            h_t = T.tanh(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W_rec) + self.b_hid)
            y_t = T.nnet.softmax(T.dot(h_t, self.W_out) + self.b_out)
            return [h_t, y_t]

        X = T.tensor3() # batch of sequence of vector
        Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null) 
        if single_output:
            Y = T.matrix() 
        else:
            Y = T.tensor3()
        h0 = shared(np.zeros(shape=(batch_size,self.n_hid), dtype=dtype)) # initial hidden state                 
        lr = shared(np.cast[dtype](lr))
        
        [h_vals, y_vals], _ = theano.scan(fn=step,        
                                          sequences=X.dimshuffle(1,0,2),
                                          outputs_info=[h0, None])

        if single_output:
            self.output = y_vals[-1]            
        else:
            self.output = y_vals.dimshuffle(1,0,2)
        
        cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
        nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))     
        mse = T.mean((self.output - Y) ** 2)

        cost = 0
        if cost_function == 'mse':
            cost = mse
        elif cost_function == 'cxe':
            cost = cxe
        else:
            cost = nll        

        gparams = T.grad(cost, self.params)
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * lr
        
        self.loss = theano.function(inputs = [X, Y], outputs = cost)
        self.train = theano.function(inputs = [X, Y], outputs = cost, updates=updates)
        self.predictions = theano.function(inputs = [X], outputs = self.output)
        self.debug = theano.function(inputs = [X, Y], outputs = [X.shape, Y.shape, y_vals.shape, self.output.shape])
