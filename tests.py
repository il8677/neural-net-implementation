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

    def test_ff_dhdw(self):
        ffl = FeedforwardLayer(3,3)

        # test dhdw
        X = np.array([2,3,4], np.float64)
        ffl.W = np.array([
            [-2, -3, -4],
            [2, 3, 4],
            [-2, -3, -4]
        ], np.float64)
        f1 = ffl.propogate(X)
        ffl.W += self.epsilon
        f2 = ffl.propogate(X)
        ffl.W -= self.epsilon

        ndf = (f2 - f1)/(self.epsilon)
        sdf = ffl.dhdw()

        diff = np.abs(sdf - ndf)

        self.assertLess(np.mean(diff), self.acceptableMargin)


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


if __name__=="__main__":
    unittest.main()