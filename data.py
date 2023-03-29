from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np

def getData():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    imageSize = train_X.shape[1]
    inputSize = imageSize * imageSize
    
    # Normalize and reshape to 1d
    train_X = np.reshape(train_X, [-1, inputSize])
    train_X = train_X.astype('float32') / 255
    test_X = np.reshape(test_X, [-1, inputSize])
    test_X = test_X.astype('float32') / 255

    return train_X, train_y, train_X, test_y

