import numpy as np
from rnnautodiff import *

xdata = np.random.randn(100)
ydata = xdata * 3

model = FeedforwardLayer(1, 1)

Trainer.train(model, xdata, ydata, MeanSquareError(), 100)

print("==============\nTraining done")
print(model.propogate(3))