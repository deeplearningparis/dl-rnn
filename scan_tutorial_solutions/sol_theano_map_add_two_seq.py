arr1 = T.fvector()
arr2 = T.fvector()

expression, updates = theano.map(add, sequences=[arr1, arr2])
f_add = theano.function([arr1, arr2], expression)