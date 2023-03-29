import data
import os.path
import pickle
from rnnautodiff import *

#loading the dataset
train_X, train_y, test_X, test_y = data.getData()

#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

def train():
    Layer.alpha = 0.1

    model = FeedforwardLayer(784, 800).next(Sigmoid())\
            .next(FeedforwardLayer(800, 10)).next(Sigmoid()).end()

    Trainer.train(model, train_X, train_y, MeanSquareError(), 20, 1, 1)

    model.save()

if __name__=="__main__":
    if os.path.isfile("model.w"):
        # load model
        with open("model.w", "rb") as f:
            model = pickle.load(f)

        totalCorrect = 0

        for i in range(test_X.shape[0]-1):
            ans = model.propogate(test_X[i])
            ans[ans == np.max(ans)] = 1
            ans[ans != np.max(ans)] = 0
            if np.equal(ans.all(), test_y[i].all()): totalCorrect += 1

        print(f"model.w accuracy: {totalCorrect/test_X.shape[0]} ({totalCorrect}/{test_X.shape[0]})")
    else:
        train()