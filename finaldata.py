import data
import os.path
import pickle
from rnnautodiff import *
import argparse

#loading the dataset
train_X, train_y, test_X, test_y = data.getData()

#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

def train(alpha, epochs, filename):
    Layer.alpha = alpha

    model = FeedforwardLayer(784, 800).next(Sigmoid())\
            .next(FeedforwardLayer(800, 10)).next(Sigmoid()).end()

    Trainer.train(model, train_X, train_y, MeanSquareError(), epochs, lossesout=filename+".csv", batchsize=1)

    model.save(filename)

def evaluate(filename):
    if os.path.isfile(filename):
        # load model
        with open(filename, "rb") as f:
            model = pickle.load(f)

        totalCorrect = 0

        for i in range(test_X.shape[0]-1):
            ans = model.propogate(test_X[i])

            # Convert to one hot classification
            ansoh = np.copy(ans)
            ansoh[ans == np.max(ans)] = 1
            ansoh[ans != np.max(ans)] = 0

            if np.array_equal(ansoh, test_y[i]): totalCorrect += 1

        print(f"{filename} accuracy: {totalCorrect/test_X.shape[0]} ({totalCorrect}/{test_X.shape[0]})")

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="MNIST Trainer")

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-ev", "--evaluate", action="store_true")
    parser.add_argument("-f", "--filename", default=None)
    parser.add_argument("-a", "--learningrate", default=0.1)
    parser.add_argument("-e", "--epochs", default=20)

    args = parser.parse_args()

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2}".format(x)})
   
    if args.filename == None:
        args.filename = f"model-{args.learningrate}-{args.epochs}.w"

    if args.evaluate:
        evaluate(args.filename)
    else:
        
        train(args.learningrate, args.epochs, args.filename)