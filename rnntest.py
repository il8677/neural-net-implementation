import numpy as np
from rnnautodiff import *

xdata = np.array([1,2,3,4,5,6])
ydata = 2 * xdata

model = FeedforwardLayer(1, 1)

Trainer.train(model, xdata, ydata, MeanSquareError(), 120)

print("==============\nTraining done")
print(model.propogate(100))