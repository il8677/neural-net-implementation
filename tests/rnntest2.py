import numpy as np
from rnnautodiff import *

steps = np.linspace(0, np.pi*2, 100, dtype = np.float32)
# the data type is float32 beacuse of converting numpy value to float Tensor

x_np = np.sin(steps)
y_np = np.cos(steps)

model = RecurrentLayer(1, 10).next(RecurrentLayer(10, 1)).end()

Trainer.trainRNN_MTM(model, x_np, y_np, MeanSquareError())

#y_pred = model.propogateRange(x_np, False).squeeze()

pass