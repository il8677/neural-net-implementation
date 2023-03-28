from rnnautodiff import *
import numpy as np
import unittest
'''
class StaticTests(unittest.TestCase):
    def test_ff_deriv11(self):
        ffl = FeedforwardLayer(1, 1)

        ffl.W = np.array([[2]])
        ffl.propogate([3])

        correctAns = [[3]]
        np.testing.assert_array_equal(ffl.dhdw(), correctAns)
        #self.assertTrue(np.array_equal(ffl.dhdw(), correctAns))


    def test_ff_deriv_5in(self):
        ffl = FeedforwardLayer(5, 1)

        ffl.W = np.array([[1], [2], [3], [4], [5]])
        ffl.propogate([10,20,30,40,50])

        correctAns = np.array([[10], [20], [30], [40], [50]])
        np.testing.assert_array_equal(ffl.dhdw(), correctAns)

    def test_ff_deriv_5out(self):
        ffl = FeedforwardLayer(1, 5)

        ffl.W = np.array([[1,2,3,4,5]])
        ffl.propogate([10])

        correctAns = np.array([[10, 10, 10, 10, 10]])
        np.testing.assert_array_equal(ffl.dhdw(), correctAns)

    def test_ff_deriv_mat(self):
        ffl = FeedforwardLayer(2, 3)

        ffl.W = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        ffl.propogate([10, 20])

        correctAns = np.array([
            [10,10,10],
            [20,20,20]
        ])
        np.testing.assert_array_equal(ffl.dhdw(), correctAns)
'''
        

class NumericalChecks(unittest.TestCase):
    epsilon = 1e-7
    acceptableMargin = 1e-7

    def test_ff_dhdi(self):
        ffl = FeedforwardLayer(3,3)

        # test dhdw
        X = np.array([2,3,4], np.float64)
        ffl.W = np.array([
            [-2, -3, -4],
            [2, 3, 4],
            [5, 6, 7]
        ], np.float64)
        
        ndf = np.zeros_like(ffl.W)

        for i in np.ndindex(X.shape):
            f1 = ffl.propogate(X)
            X[i] += self.epsilon
            f2 = ffl.propogate(X)
            X[i] -= self.epsilon

            ndf[i] = (f2 - f1)/(self.epsilon)
        
        sdf = ffl.dhdi()
        diff = np.abs(sdf - ndf)
        self.assertLess(np.mean(diff), self.acceptableMargin)
    
    def test_ff_dhdi2(self):
        ffl = FeedforwardLayer(3,4)

        # test dhdw
        X = np.array([2,3,4], np.float64)
        ffl.W = np.array([
            [-2, -3, -4, -5],
            [2, 3, 4, 5],
            [5, 6, 7, 8],
        ], np.float64)
        
        ndf = np.zeros_like(ffl.W)

        for i in np.ndindex(X.shape):
            f1 = ffl.propogate(X)
            X[i] += self.epsilon
            f2 = ffl.propogate(X)
            X[i] -= self.epsilon

            ndf[i] = (f2 - f1)/(self.epsilon)
        
        sdf = ffl.dhdi()
        diff = np.abs(sdf - ndf)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_ff_dhdi3(self):
        ffl = FeedforwardLayer(4,3)

        # test dhdw
        X = np.array([2,3,4,5], np.float64)
        ffl.W = np.array([
            [-2, -3, -4],
            [2, 3, 4],
            [5, 6, 7],
            [5, 6, 7]
        ], np.float64)
        
        ndf = np.zeros_like(ffl.W)

        for i in np.ndindex(X.shape):
            f1 = ffl.propogate(X)
            X[i] += self.epsilon
            f2 = ffl.propogate(X)
            X[i] -= self.epsilon

            ndf[i] = (f2 - f1)/(self.epsilon)
        
        sdf = ffl.dhdi()
        diff = np.abs(sdf - ndf)
        self.assertLess(np.mean(diff), self.acceptableMargin)
    


    def test_mse(self):
        mse = MeanSquareError()
        pred = np.array([1,2,3,4,5], np.float64)
        true = np.array([3, 5, 8, 12, 17], np.float64)
        
        ndl = np.zeros_like(pred)

        for i in np.ndindex(ndl.shape):
            f1 = mse.getError(pred, true)
            pred[i] += self.epsilon
            f2 = mse.getError(pred, true)
            pred[i] -= self.epsilon
            ndl[i] = (f1-f2)/self.epsilon

        sdl = mse.getDeriv(pred, true)

        diff = np.abs(sdl - ndl)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_bce(self):
        bce = BinaryCrossEntropy()
        pred = np.array([1,0,1,1,0], np.float64)
        true = np.array([0,1,1,0,1], np.float64)
        
        ndl = np.zeros_like(pred)

        for i in np.ndindex(ndl.shape):
            f1 = bce.getError(pred, true)
            pred[i] += self.epsilon
            f2 = bce.getError(pred, true)
            pred[i] -= self.epsilon
            ndl[i] = (f1-f2)/self.epsilon

        sdl = bce.getDeriv(pred, true)

        diff = np.abs(sdl - ndl)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_sigmoid_dhdi(self):
        s = Sigmoid()
        i = np.asarray([1,2,3,4,5], np.float64)

        ndl = np.zeros_like(i)

        f1 = s.propogate(i)
        i += self.epsilon
        f2 = s.propogate(i)
        i -= self.epsilon

        ndh = (f2-f1)/self.epsilon
        sdh = s.backwards(1)

        diff = np.abs(ndh - sdh)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_dldw(self):
        mse = MeanSquareError()
        ff1 = FeedforwardLayer(2, 3)
        sig = Sigmoid()

        ff1.next(sig).end()

        data = [1,2]
        actual = [0.5, 0.75, 0.23]

        ndl = np.zeros_like(ff1.W, np.float64)

        for iy, ix in np.ndindex(ff1.W.shape):
            pred = ff1.propogate(data)
            ff1.W[iy, ix] += self.epsilon
            pred2 = ff1.propogate(data)
            ff1.W[iy, ix] -= self.epsilon

            f1 = mse.getError(pred, actual)
            f2 = mse.getError(pred2, actual)

            ndl[iy, ix] = (f1 - f2)/self.epsilon

        dlds = sig.backwards(mse.getDeriv(pred, actual))
        sdl = ff1.dldw(dlds)

        diff = np.abs(sdl - ndl)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_dldw2(self):
        mse = MeanSquareError()
        ff1 = FeedforwardLayer(2, 3)
        sig1 = Sigmoid()
        ff2 = FeedforwardLayer(3, 2)
        sig2 = Sigmoid()

        ff1.next(sig1).next(ff2).next(sig2).end()

        data = [1,2]
        actual = [0.5, 0.75]

        ndl = np.zeros_like(ff1.W, np.float64)

        for iy, ix in np.ndindex(ff1.W.shape):
            pred = ff1.propogate(data)
            ff1.W[iy, ix] += self.epsilon
            pred2 = ff1.propogate(data)
            ff1.W[iy, ix] -= self.epsilon

            f1 = mse.getError(pred, actual)
            f2 = mse.getError(pred2, actual)

            ndl[iy, ix] = (f1 - f2)/self.epsilon

        dlds = sig1.backwards(
               ff2.backwards(
               sig2.backwards(
               mse.getDeriv(pred, actual))))
        
        sdl = ff1.dldw(dlds)

        diff = np.abs(sdl - ndl)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_tanh(self):
        t = Tanh()

        f1 = t.propogate(2.0)
        f2 = t.propogate(2.0 + self.epsilon)

        ndt = (f2 - f1) / self.epsilon
        sdt = t.backwards(1)

        diff = np.abs(sdt - ndt)
        self.assertLess(np.mean(diff), self.acceptableMargin)

'''

class RNN_NumericalChecks(unittest.TestCase):
    epsilon = 1e-14
    acceptableMargin = 1e-7

    def test_recurrent(self):
        rnn = RecurrentLayer(1,1)

        rnn.W1[0][0] = 3
        rnn.W2[0][0] = 4

        r1 = rnn.propogate(1)
        r2 = rnn.propogate(2)
        r3 = rnn.propogate(3)

        self.assertEqual(r1, 3*1)
        self.assertEqual(r2, r1 * 4 + 2 * 3)
        self.assertEqual(r3, r2 * 4 + 3 * 3)

    def test_dldw_small(self):
        mse = MeanSquareError()
        rnn = RecurrentLayer(1, 1)

        rnn.W1 = np.ones_like(rnn.W1) * 1
        rnn.W2 = np.ones_like(rnn.W2) * 1

        data = np.asarray([[[1], [2]]], np.float32)
        actual = np.asarray([[[2], [4]]], np.float32)

        ndl = np.zeros_like(rnn.W1, np.float64)

        for iy, ix in np.ndindex(rnn.W1.shape):
            pred = rnn.propogateRange(data)
            rnn.clear()
            rnn.W1[iy, ix] += self.epsilon
            pred2 = rnn.propogateRange(data)
            rnn.clear()
            rnn.W1[iy, ix] -= self.epsilon

            f1 = mse.getError(pred, actual)
            f2 = mse.getError(pred2, actual)

            ndl[iy, ix] = (f1 - f2)/self.epsilon

        pred = rnn.propogateRange(data)
        #sdl1 = rnn.dldw1(mse.getDeriv(pred[0][0], actual[0][0]), 0)
        #sdl2 = rnn.dldw1(mse.getDeriv(pred[0][1], actual[0][1]), 1)
        
        sdl1 = rnn.dldw1(pred[0][0], 0)
        sdl2 = rnn.dldw1(pred[0][1], 1)
        sdl = sdl1 + sdl2

        diff = np.abs(sdl - ndl)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_dldw1(self):
        mse = MeanSquareError()
        rnn = RecurrentLayer(2, 3)

        rnn.W1 = np.ones_like(rnn.W1) * 1
        rnn.W2 = np.ones_like(rnn.W2) * 1

        data = np.asarray([[[1,2], [3,4]]], np.float32)
        actual = np.asarray([[[0.5, 0.75, 0.23], [0.35, 0.12, 0.34]]], np.float32)

        ndl = np.zeros_like(rnn.W1, np.float64)

        for iy, ix in np.ndindex(rnn.W1.shape):
            pred = rnn.propogateRange(data)
            rnn.clear()
            rnn.W1[iy, ix] += self.epsilon
            pred2 = rnn.propogateRange(data)
            rnn.clear()
            rnn.W1[iy, ix] -= self.epsilon

            f1 = mse.getError(pred, actual)
            f2 = mse.getError(pred2, actual)

            ndl[iy, ix] = (f1 - f2)/self.epsilon

        pred = rnn.propogateRange(data)
        sdl1 = rnn.dldw1(mse.getDeriv(pred[0][0], actual[0][0]), 0)
        sdl2 = rnn.dldw1(mse.getDeriv(pred[0][1], actual[0][1]), 1)
        sdl = sdl1 + sdl2

        diff = np.abs(sdl - ndl)
        self.assertLess(np.mean(diff), self.acceptableMargin)

    def test_dhdw(self):
        return
        rnn = RecurrentLayer(2, 3)

        data = np.asarray([[[1,2], [3,4]]], np.float32)
        actual = np.asarray([[[0.5, 0.75, 0.23], [0.35, 0.12, 0.34]]], np.float32)

        dhdw = np.zeroes(())
'''

if __name__=="__main__":
    unittest.main()