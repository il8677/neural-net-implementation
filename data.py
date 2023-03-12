import numpy

def getx():
    return numpy.random.randint(20, size=(100, 10,1)).astype(numpy.float32) % 2

def gety(x):
    y = numpy.zeros_like(x)
    

    for l in x[0]:
        c = 0
        acc = 0
        for i in l:
            acc += i
            l[c] = acc % 2
            c+=1

    return y 
