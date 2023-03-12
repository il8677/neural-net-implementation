import numpy
from data import *

xdata = getx()
x_in = []
truth = []
y_out = gety(xdata)
yhat_out = numpy.zeros_like(y_out[0], dtype=numpy.float64)

hiddenLayerSize = 10

W1 = numpy.random.randn(hiddenLayerSize)
W2 = numpy.random.randn(hiddenLayerSize)
W3 = numpy.random.randn(hiddenLayerSize)
W4 = numpy.random.randn()

learning_rate = 0.05

# Per timestep functions
def zt(t):
    if t < 0:
        return 0
    return numpy.dot(x_in[t], W1) + numpy.dot(zt(t-1), W2) + W4

def dztw1(t):
    return x_in[t]

def dztw2(t):
    return zt(t-1)

def dztw4(t):
    return 1

def ht(x):
    return numpy.tanh(x)

def dht(x):
    return 1 - numpy.power(numpy.tanh(x), 2)

def yt(x):
    return numpy.dot(x, W3)

def dyt(x):
    return x

def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

def dyhat(x):
    return sigmoid(x) * (1 - sigmoid(x))

def yhat(x):
    return sigmoid(x)

def error(y_pred, y_true, debug = False):
    eps = 1e-15  # small constant to avoid division by zero
    y_pred = numpy.clip(y_pred, eps, 1 - eps)  # clip predicted probabilities to avoid log(0)
    loss = -y_true * numpy.log(y_pred) - (1 - y_true) * numpy.log(1 - y_pred)
    
    if debug:
        print(y_true)
        print(y_pred)
        print(loss)
    return numpy.mean(loss)

# Forward propagation
def forward_prop():
    global yhat_out
    for i in range(len(x_in)):
        s1 = zt(i)
        s2 = ht(s1)
        s3 = yt(s2)
        s4 = yhat(s3)
        yhat_out[i] = s4
        
        #print(f"{x_in[i]} -> {s1} -> {s2} -> {s3} -> {s4}  ({y_out[i]})")

# Backpropagation
def back_prop():
    global W1, W2, W3, W4
    dW1, dW2, dW3, dW4 = 0, 0, 0, 0
    for i in range(len(x_in)-1, -1, -1):
        # Output layer delta
        delta4 = (yhat_out[i] - truth[i])
        #  dW3 += delta4 * ht(W1 * x_in[i] + W2 * zt(i-1))
        dydht = dyhat(yt(ht(zt(i)))) * dyt(ht(zt(i)))
        dhtdz = dht(zt(i))
        dztdw1 = dztw1(i)
        dztdw2 = dztw2(i)
        dztdw4 = dztw4(i)

        dW3 += delta4 * dydht

        # Hidden layer delta
        delta3 = delta4
        dW1 += delta3 * dydht * dhtdz * dztdw1
        dW2 += delta3 * dydht * dhtdz * dztdw2
        dW4 += delta3 * dydht * dhtdz * dztdw4

    # Update weights
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    W3 -= learning_rate * dW3
    W4 -= learning_rate * dW4

    return dW1, dW2, dW3

if __name__ == "__main__":
    print(f"W1: {W1} W2: {W2} W3: {W3}")
    losses = []
    WS = []
    for epoch in range(100000):
        for dataexample, yexample in zip(xdata[0], y_out[0]):
            x_in = dataexample
            truth = y_out
            forward_prop()
            d1, d2, d3 = back_prop()

        loss = error(yhat_out, truth, False)

        losses.append(loss)

        if epoch % 1000 == 0:
            #x_in = getx()
            #y_out = gety(x_in)
            #print(f"W1: {W1} W2: {W2} W3: {W3} W4: {W4}")
            #print(f"D1: {d1} D2: {d2} D3: {d3}")
            print(yhat_out)
            print("Epoch {}, Loss: {}".format(epoch, loss))
            print("=="*35)

    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.show()