arr = T.fvector()  # to be removed
expression, updates = theano.map(square, sequences=arr)

f_square_arr = theano.function([arr], expression)