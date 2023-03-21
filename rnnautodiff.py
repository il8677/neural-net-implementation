import numpy as np

# Used https://www.youtube.com/watch?v=UpLtbV4L6PI as reference

d_debug = False

# Glorot Beningo
def init_weights(shape):
    # Calculate the range of the uniform distribution
    limit = np.sqrt(6 / np.sum(shape))
    
    # Initialize the weights with values drawn from a uniform distribution
    weights = np.random.uniform(-limit, limit, shape)
    
    return weights

def clip_gradient(grad, max_norm):
    return grad
    grad_norm = np.linalg.norm(grad)
    if grad_norm > max_norm:
        grad = grad * (max_norm / grad_norm)
    return grad

class Layer():
    alpha = 0.01

    def getHeader():
        return "=" * 36

    def __init__(self):
        self.prev = None
        self.n = None
        self.input = []
        self.H = []

    def propogate(self, X):
        if self.n:
            return self.n.propogate(self.H)
        else:
            return self.H
        
    def propogateRange(self, X):
        outputsize = self.propogate(X[0]).shape[0]
        output = np.zeros((X.shape[0], outputsize))
        c = 0
        for i in X:
            output[c] = self.propogate(X[c])
            
            c += 1

        return output
    
    def backpropogateRange(self, accs, X):
        for acc, x in zip(accs, X):
            self.end().propogate(x)
            self.backpropogate(acc)

    def next(self, n):
        self.n = n
        n.prev = self

        return n
    
    def end(self):
        if self.prev:
            return self.prev.end()
        else:
            return self
        
    def tail(self):
        if self.n:
            return self.n.tail()
        else:
            return self

    # Where h is the output and l is the input function, i are the inputs
    # Calculate dl/di given dl/dh
    def backwards(self, dldh):
        pass

    def backpropogate(self, dldh=1):
        dldh = self.backwards(dldh)
        if self.prev:
            self.prev.backpropogate(dldh)

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
    # dh/dxj = sum d(wij * xj)/dxj
    # dh/dxj = sum (wij)
    #
    # Vectorized:
    # dh/dw Matrix full of xj
    # dh/dx Matrix W summed row

    def __init__(self, inputsize, size=10):
        super().__init__()
        self.W = init_weights((inputsize, size))
        self.H = np.zeros(size)
        self.input = np.zeros(inputsize)

    def propogate(self, X):
        X = np.asarray(X)
        assert X.size == self.W.shape[0]
        self.input = X
        self.H = np.dot(X, self.W)
        return super().propogate(X)
    
    def backpropogate(self, dldh=1):
        dW = clip_gradient(Layer.alpha * self.dldw(dldh), 2.0)
        assert dW.shape == self.W.shape
        self.W += dW
        if d_debug: 
            print(f"{self.input} -> {self.H} Updating weight by {Layer.alpha} * {dldh} * {self.dhdw()} = {dW}")
        super().backpropogate(dldh)


    def backwards(self, dldh):
        return np.dot(self.dhdi(), dldh).flatten()
    
    def dhdi(self):
        return self.W

    def dldw(self, dldh):
        #return self.input * np.ones_like(self.W)
        return np.outer(self.input.T, dldh)

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
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def propogate(self, X):
        self.input = X
        self.H = self.sigmoid(X)
        return super().propogate(X)

    def backwards(self, dldh):
        return dldh * self.dsigmoid(self.input)
    
    def print(self):
        print(Layer.getHeader())
        print("Sigmoid Layer")
        print(self.input)
        print(self.H)

        super().print()

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def tanh(self, X):
        return np.tanh(X)

    def dtanh(self, X):
        return 1 - np.power(np.tanh(X), 2)

    def propogate(self, X):
        self.input = X
        self.H = self.tanh(X)
        return super().propogate(X)

    def backwards(self, dldh):
        return dldh * self.dtanh(self.input)
    
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
    # E(t,p) = sum i < k (ti - pi)**2 / k
    # Scalar Derivative
    # de/dtj = sum i < k 1/k d((ti-pi)**2)/dtj
    #        = 1/k d((tj-pj)**2)/dtj
    #        = 1/k d(u**2)/du  (tj-pj)/dtj
    #        = 1/k 2u * 1
    #        = 1/k 2(tj-pj)
    # Vectorize
    # foreach i < k 2(tj-pj)/k
    def MSE(self, pred, true):
        return np.mean((true - pred)**2)

    def getError(self, pred, true):
        return self.MSE(pred, true)
    
    def getDeriv(self, pred, true):
        return 2 * (true - pred) / pred.size 
    

class Trainer:
    def train(model: Layer, x, y, error: Error, epochs=100, printinterval=1, batchsize=32):
        batchsize = batchsize if batchsize < x.shape[0] else x.shape[0]
        batchcount = int(np.ceil(x.shape[0] / batchsize))
        for epoch in range(epochs):
            losses = []
            for i in range(batchcount):

                start_idx = i * batchsize
                end_idx = min(start_idx + batchsize, x.shape[0])
                batch_X = x[start_idx:end_idx]
                batch_Y = y[start_idx:end_idx]

                pred = model.propogateRange(batch_X).squeeze()
                losses.append(np.mean(error.getError(pred, batch_Y)))

                model.tail().backpropogateRange(error.getDeriv(pred, batch_Y), batch_X)
                if d_debug: 
                    input()
                    print(Layer.getHeader())

            if epoch % printinterval == 0:
                print(f"Epoch {epoch}: {np.mean(losses)}")