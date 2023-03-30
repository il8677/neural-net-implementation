
import numpy
from data import *
import keras as keras
from keras import layers
import os

train_X, train_y, test_X, test_y = getData()

if not len(os.sys.argv) == 2:
    model = keras.Sequential()
    model.add(layers.Input(shape=(784)))
    model.add(layers.Dense(800, activation='sigmoid', use_bias=False))
    model.add(layers.Dense(10, activation='sigmoid', use_bias=False))

    model.summary()

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.1), metrics=[keras.metrics.CategoricalAccuracy()])
    history = model.fit(train_X, train_y, epochs=20, batch_size=1)
    model.save("kerasmodel.w")
    with open("kerashistory.h", "w") as f:
        f.write(history.history)
else:
    model = keras.models.load_model(os.sys.argv[1])
    model.compile(optimizer=model.optimizer,
                        loss=model.loss,
                        metrics=[keras.metrics.CategoricalAccuracy()])

loss, accuracy = model.evaluate(test_X, test_y, batch_size=1)
print("Loss, accuracy:", loss, accuracy)

