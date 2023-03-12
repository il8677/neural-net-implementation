
import numpy
from data import *
import keras as keras
from keras import layers

x_in = getx()
y_out = gety(x_in)
yhat_out = numpy.zeros_like(y_out)

model = keras.Sequential()
model.add(layers.SimpleRNN(1))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='SGD')

model.fit(x_in, y_out, epochs=20)
