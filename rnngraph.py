import numpy as np
from rnnautodiff import *

xdata = np.random.randn(100)
ydata = xdata * 3

model = FeedforwardLayer(1, 1)
lossFunc = MeanSquareError()

# Get the loss as the weight changes
x = np.linspace(-10, 10, 300)

y1 = []
y2 = []
y3 = []
for w in x:
    model.W = [[w]]

    pred = model.propogate(3)
    error = lossFunc.getError(pred, 3*3)
    derror = lossFunc.getDeriv(pred.item(), 3*3)
    
    y1.append(error)
    y2.append(derror)
    y3.append(model.dhdw().item())

import matplotlib.pyplot as plt

plt.plot(x, y1)
plt.plot(x, y2)
plt.gca().set_ylim(bottom=0)
plt.show()

plt.plot(x, y3)
plt.gca().set_ylim(bottom=0)
plt.show()