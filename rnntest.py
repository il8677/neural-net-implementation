import numpy as np
from rnnautodiff import *

xdata = np.array([1,2,3,40])
ydata = 2 * Sigmoid().sigmoid(xdata)

model = FeedforwardLayer(1, 10).next(Sigmoid()) \
  .next(FeedforwardLayer(10, 1)).end()

model.print()

Trainer.train(model, xdata, ydata, MeanSquareError(), 120)

print("==============\nTraining done")
print(model.propogate(100))
model.print()
