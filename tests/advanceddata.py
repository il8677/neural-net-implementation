import pandas as pd
from rnnautodiff import *

df = pd.read_csv("X_train.tsv", sep="\t").to_numpy()
df2 = pd.read_csv("y_train.tsv", sep="\t").to_numpy().squeeze()

Layer.alpha = 0.01

model = FeedforwardLayer(5, 12).next(Sigmoid())\
    .next(FeedforwardLayer(12, 8)).next(Sigmoid()) \
    .next(FeedforwardLayer(8,1))\
    .end()

Trainer.train(model, df, df2, MeanSquareError(), 1000)