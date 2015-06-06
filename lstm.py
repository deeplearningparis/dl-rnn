import numpy as np
import theano
import theano.tensor as T
from theano import shared 
from collections import OrderedDict
from init import init_weight

dtype=T.config.floatX

print "loaded lstm.py"

class Lstm:
    def __init__(self, n_in, n_lstm, n_out, lr=0.05, single_output=True, output_activation=T.nnet.softmax, cost_function='nll'):        
        self.n_in = n_in
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_xi = init_weight((self.n_in, self.n_lstm),'W_xi') 
        self.W_hi = init_weight((self.n_lstm, self.n_lstm),'W_hi', 'svd') 
        self.W_ci = init_weight((self.n_lstm, self.n_lstm),'W_ci', 'svd') 
        self.b_i = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
        self.W_xf = init_weight((self.n_in, self.n_lstm),'W_xf') 
        self.W_hf = init_weight((self.n_lstm, self.n_lstm),'W_hf', 'svd') 
        self.W_cf = init_weight((self.n_lstm, self.n_lstm),'W_cf', 'svd') 
        self.b_f = shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_lstm)))
        self.W_xc = init_weight((self.n_in, self.n_lstm),'W_xc') 
        self.W_hc = init_weight((self.n_lstm, self.n_lstm),'W_hc', 'svd') 
        self.b_c = shared(np.zeros(n_lstm, dtype=dtype))
        self.W_xo = init_weight((self.n_in, self.n_lstm),'W_xo') 
        self.W_ho = init_weight((self.n_lstm, self.n_lstm),'W_ho', 'svd') 
        self.W_co = init_weight((self.n_lstm, self.n_lstm),'W_co', 'svd') 
        self.b_o = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
        self.W_hy = init_weight((self.n_lstm, self.n_out),'W_hy') 
        self.b_y = shared(np.zeros(n_out, dtype=dtype))
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, 
                       self.W_xf, self.W_hf, self.W_cf, self.b_f, 
                       self.W_xc, self.W_hc, self.b_c, 
                       self.W_ho, self.W_co, self.W_co, self.b_o, 
                       self.W_hy, self.b_y]
                

        def step_lstm(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c) 
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co)  + self.b_o)
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b_y) 
            return [h_t, c_t, y_t]

        X = T.matrix() # batch of sequence of vector
        Y = T.matrix() # batch of sequence of vector (should be 0 when X is not null) 
        if single_output:
            Y = T.vector() 
        h0 = shared(np.zeros(shape=self.n_lstm, dtype=dtype)) # initial hidden state         
        c0 = shared(np.zeros(shape=self.n_lstm, dtype=dtype)) # initial hidden state         
        lr = shared(np.cast[dtype](lr))
        
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=step_lstm,        
                                          sequences=X,
                                          outputs_info=[h0, c0, None])

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
        self.debug = theano.function(inputs = [X, Y], outputs = [X.shape, Y.shape, y_vals.shape, cost.shape])


class LstmMiniBatch:
    def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, single_output=True, output_activation=T.nnet.softmax, cost_function='nll'):        
        self.n_in = n_in
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_xi = init_weight((self.n_in, self.n_lstm),'W_xi') 
        self.W_hi = init_weight((self.n_lstm, self.n_lstm),'W_hi', 'svd') 
        self.W_ci = init_weight((self.n_lstm, self.n_lstm),'W_ci', 'svd') 
        self.b_i = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
        self.W_xf = init_weight((self.n_in, self.n_lstm),'W_xf') 
        self.W_hf = init_weight((self.n_lstm, self.n_lstm),'W_hf', 'svd') 
        self.W_cf = init_weight((self.n_lstm, self.n_lstm),'W_cf', 'svd') 
        self.b_f = shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_lstm)))
        self.W_xc = init_weight((self.n_in, self.n_lstm),'W_xc') 
        self.W_hc = init_weight((self.n_lstm, self.n_lstm),'W_hc', 'svd') 
        self.b_c = shared(np.zeros(n_lstm, dtype=dtype))
        self.W_xo = init_weight((self.n_in, self.n_lstm),'W_xo') 
        self.W_ho = init_weight((self.n_lstm, self.n_lstm),'W_ho', 'svd') 
        self.W_co = init_weight((self.n_lstm, self.n_lstm),'W_co', 'svd') 
        self.b_o = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
        self.W_hy = init_weight((self.n_lstm, self.n_out),'W_hy') 
        self.b_y = shared(np.zeros(n_out, dtype=dtype))
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, 
                       self.W_xf, self.W_hf, self.W_cf, self.b_f, 
                       self.W_xc, self.W_hc, self.b_c, 
                       self.W_ho, self.W_co, self.W_co, self.b_o, 
                       self.W_hy, self.b_y]
                

        def step_lstm(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c) 
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co)  + self.b_o)
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b_y) 
            return [h_t, c_t, y_t]

        X = T.tensor3() # batch of sequence of vector
        Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null) 
        h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state         
        c0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state         
        lr = shared(np.cast[dtype](lr))
        
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=step_lstm,        
                                          sequences=X.dimshuffle(1,0,2),
                                          outputs_info=[h0, c0, None])

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
        
        self.loss = theano.function(inputs = [X, Y], outputs = [cxe, mse, cost])
        self.train = theano.function(inputs = [X, Y], outputs = cost, updates=updates)
        self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2))
        self.debug = theano.function(inputs = [X, Y], outputs = [X.shape, Y.shape, y_vals.shape, cxe.shape])
