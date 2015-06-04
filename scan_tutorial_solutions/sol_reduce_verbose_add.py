def verbose_add(a, b): 
    print "a={}, b={}".format(str(a), str(b))
    return a + b
reduce(verbose_add, map(lambda x: [x], np.arange(10)), [100])
