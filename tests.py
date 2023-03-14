from rnnautodiff import *
import numpy as np
import unittest

def testModelDerivatives():
    epsilon = 1e-15

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
        

class NumericalChecks(unittest):
    pass

if __name__=="__main__":
    unittest.main()