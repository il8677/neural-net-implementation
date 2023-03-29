import numpy as np
import pickle

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

    def propogate_forward(self, X, H):
        if self.n:
            return self.n.propogate(H)
        else:
            return self.H
        
    def propogate(self, X):
        return self.propogate_forward(X, self.H)
    
    # Propogates multiple values through the network
    def propogateRange(self, X, doClear=True):
        output = []
        c = 0
        for i in X:
            if doClear: self.clear()
            # If it's recurrent data, propogate each one, not clearing
            # Not clearing means that the recurrent layer will recur
            if type(i) == np.ndarray and i.ndim != 1:
                output.append(self.propogateRange(i, False))
            else:
                output.append(self.propogate(X[c]))
            c += 1

        return np.stack(output)
    
    def backpropogateRange(self, accs, X):
        for acc, x in zip(accs, X):
            self.propogateRange([x])
            self.prop_backprop(acc)
            self.clear()

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

    # Propogates a backpropogation call through the network
    def prop_backprop(self, dldh):
        if self.n:
            dldh = self.n.prop_backprop(dldh)
        
        return self.backpropogate(dldh)

    def backpropogate(self, dldh=1):
        dldh = self.backwards(dldh)
        return dldh

    def print(self):
        if self.n:
            self.n.print()

    def save(self, filename="model.w"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def clear(self):
        pass

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
        return super().backpropogate(dldh)


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

class RecurrentLayer(Layer):
    def __init__(self, inputsize, size=10):
        super().__init__()
        self.W1 = init_weights((inputsize, size))
        self.W2 = init_weights((size, size))
        self.H = np.zeros(size)
        self.outs = []
        self.inputs = []
        self.input = np.zeros(inputsize)
        self.tau = 0

    def propogate(self, X):
        X = np.asarray(X)
        assert X.size == self.W1.shape[0]
        self.input = X
        self.inputs.append(X)
        self.tau += 1

        tcomp = np.dot(self.outs[-1], self.W2) if self.outs.__len__() else 0
        self.outs.append(np.add(np.dot(X, self.W1), tcomp))
        self.H = self.outs[-1]

        return super().propogate(X)
                
    def prop_backprop(self, dldhtau, tau=-1):
        if tau == -1: tau = self.tau-1

        if self.n:
            dldhtau = self.n.prop_backprop(dldhtau, tau)

        dldht = dldhtau.flatten()
        for t in range(tau, -1, -1):
            dldht = self.backpropogate(t, dldht).flatten()
            dldht = clip_gradient(dldht, 10)

        return self.dhdi() * dldhtau
    
    # Returns new dldh(t-1)
    def backpropogate(self, t, dldht=1):
        dtdhtm1 = self.dhtdhtm1()
        dldhtm1 = np.dot(dldht, dtdhtm1)


        # dldht * dht/dout_t-1 * dout_t/dW1
        dW1 = np.outer(self.dhdw1(t), dldht.T)
        # dldhtau * dhtau/dout_t * dout_t/dW2
        dW2 = np.outer(self.dhdw2(t), dldht.T)

        assert dW1.shape == self.W1.shape
        assert dW2.shape == self.W2.shape
        assert not np.isnan(np.sum(dW1))
        assert not np.isnan(np.sum(dW2))

        self.W1 += dW1
        self.W2 += dW2

        assert not np.isnan(np.sum(dldhtm1))
        return dldhtm1

    def backwards(self, dldh):
        return np.dot(self.dhdi(), dldh).flatten()
    
    def dhtdhtm1(self):
        # ht = W1 dot x + W2 ht-1
        return self.W2

    def dhdi(self):
        return self.W1

    def dhdw1(self, t):
        #return self.input * np.ones_like(self.W)
        return self.inputs[t].T
    
    def dhdw2(self, t):
        if t-1 < 0 or t-1 > self.outs.__len__(): 
            return np.zeros_like(self.outs[t].T)

        return self.outs[t-1].T

    def clear(self):
        self.outs = []
        self.inputs = []
        self.tau = 0
        return super().clear()

    def print(self):
        print(Layer.getHeader())
        print("Recurrent Layer")
        print(self.W1)
        print(self.W2)
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

class BinaryCrossEntropy(Error):
    def BCE(y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
        term_1 = y_true * np.log(y_pred + 1e-7)
        return -np.mean(term_0+term_1, axis=0)

    def getError(self, pred, true):
        return BinaryCrossEntropy.BCE(pred, true)
    def getDeriv(self, pred, true):
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return -true * (1/pred) - (1 - true) * (1/(pred-1))

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
    def train(model: Layer, x, y, error: Error, epochs=100, printinterval=1, batchsize=32, batchprintinterval=5):
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

                model.backpropogateRange(error.getDeriv(pred, batch_Y), batch_X)
                if d_debug: 
                    input()
                    print(Layer.getHeader())
                print(f"==> Batch {i}: {np.mean(losses[-batchprintinterval:-1])}", end="\r")

            if epoch % printinterval == 0:
                print(f"Epoch {epoch}: {np.mean(losses):.3g} {np.median(losses):.3g}")

    def trainRNN_MTM(model: RecurrentLayer, x, y, error: Error, epochs=100, printinterval=1):
        for epoch in range(epochs):
            losses = []
            pred = model.propogateRange(x, False).squeeze()
            losses.append(np.mean(error.getError(pred, y)))

            for t in range(pred.__len__()-1, -1, -1):
                loss = error.getError(pred[t], y[t])
                dl = error.getDeriv(pred[t], y[t])

                model.prop_backprop(dl, t)

            if d_debug: 
                input()
                print(Layer.getHeader())

            if epoch % printinterval == 0:
                print(f"Epoch {epoch}: {np.mean(losses)}")