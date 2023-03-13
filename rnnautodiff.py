import numpy as np

# Used https://www.youtube.com/watch?v=UpLtbV4L6PI as reference

class Layer():
    alpha = 0.01

    def getHeader():
        return "=" * 36

    def __init__(self):
        self.prev = None
        self.n = None

    def propogate(self, X):
        if self.n:
            return self.n.propogate(self.H)
        else:
            return self.H
        
    def propogateRange(self, X):
        output = np.zeros_like(X)
        c = 0
        for i in X:
            output[c] = self.propogate(i)
            
            c += 1

        return output
    
    def backpropogateRange(self, accs):
        c = 0
        for acc in accs:
            self.backpropogate(acc)

    def start(self):
        self.prev = None
        return self

    def next(self, n):
        self.n = n
        n.prev = self

        return n
    
    def end(self):
        if self.prev:
            return self.prev.end()
        else:
            return self

    # Where h is the output and l is the input function, i are the inputs
    # Calculate dl/di given dl/dh
    def backwards(self, dldh):
        pass

    def backpropogate(self, acc=1):
        acc *= self.backwards(acc)
        if self.prev:
            self.prev.backpropogate(acc)

    def print(self):
        if self.n:
            self.n.print()

class FeedforwardLayer(Layer):
    # A normal layer of a neural network
    # Has fixed input size, and size
    # h(x) = W * x
    #      = row i = sum j (wij * xj)
    # 
    # Scalar derivative:
    # dh/dwij = d(wij * xj)/dwij
    #         = xj
    #
    # dh/dxj = d(wij * xj)/dxj
    # dh/dxj = wij
    #
    # Vectorized:
    # dh/dw Matrix full of xj
    # dh/dx Matrix W

    def __init__(self, inputsize, size=10):
        super().__init__()
        self.W = np.random.randn(inputsize, size)
        self.H = np.zeros(size)
        self.input = np.zeros(inputsize)

    def propogate(self, X):
        self.input = X
        self.H = np.dot(X, self.W)
        return super().propogate(X)

    def backpropogate(self, acc=1):
        self.W -= Layer.alpha * acc

        super().backpropogate(acc)

    def backwards(self, dldh):
        return dldh * self.dhdi()
    
    def dhdi(self):
        return self.W

    def dhdw(self):
        return self.input * np.ones_like(self.W)

    def print(self):
        print(Layer.getHeader())
        print("Standard Layer")
        print(self.W)
        print(self.input)
        print(self.H)

        super().print()

class Sigmoid(Layer):
    # A sigmoid layer
    # s(x) = 1/(1 + e^-x)
    #      = sum
    # Derivative:
    # s(x) * (1 - s(x)) 
    def __init__(self):
        super().__init__()

    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))

    def dsigmoid(self, X):
        return self.sigmoid(X) - (1 - self.sigmoid(X))

    def propogate(self, X):
        self.input = X
        self.H = self.sigmoid(X)
        return super().propogate(X)

    def backwards(self, dldh):
        return dldh * self.dsigmoid(self.H)
    
    def print(self):
        print(Layer.getHeader())
        print("Sigmoid Layer")
        print(self.input)
        print(self.H)

        super().print()

class Error:
    def getError(self, pred, true):
        raise NotImplementedError()
    
    def getDeriv(self, pred, true):
        raise NotImplementedError()

class MeanSquareError(Error):
    def MSE(self, pred, true):
        return np.mean((true - pred)**2)

    def getError(self, pred, true):
        return self.MSE(pred, true)
    
    def getDeriv(self, pred, true):
        return 2 * (true - pred)
    

class Trainer:
    def train(model: Layer, x, y, error: Error, epochs=100, printinterval=1):
        for epoch in range(epochs):
            pred = model.propogateRange(x)
            loss = error.getError(pred, y)
            model.backpropogateRange(error.getDeriv(pred, y))

            if epoch % printinterval == 0:
                print(f"Epoch {epoch}: {loss}")