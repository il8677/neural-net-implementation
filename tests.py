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
        f1 = ffl.propogate(X)
        sdf = ffl.dhdi()
        X += self.epsilon
        f2 = ffl.propogate(X)
        X -= self.epsilon

        ndf = (f2 - f1)/(self.epsilon)

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


if __name__=="__main__":
    unittest.main()