from Linear import Linear
from SimpleRNN import SimpleRNN
from LSTM import LSTM
from GRU import GRU

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.model_selection import  train_test_split

import numpy as np
import matplotlib.pyplot as plt

###Build the dataset
#This is a nonlinear AND long-distance dataset
#(Actually, I will test long-distance vs. short-distance patterns)

#Define a get_label function which does the XOR for three inputs from x
def get_label(x, i1, i2, i3):
    #x = sequence
    if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
        return 1
    if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
        return 1
    return 0
#Create the X list and Y list. X is a series of random distributed values and Y is the output of get_label()
def create_data(T, i1, i2, i3):
    X, Y = [], []
    for t in range(5000):
        x = np.random.randn(T)
        y = get_label(x, i1, i2, i3)

        X.append(x)
        Y.append(y)

    X, Y = np.array(X), np.array(Y)
    N = len(Y)

    return X, Y, N
#Define a function to plot the training results
def training_results(r):
    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend
    plt.show()

    plt.plot(r.history["accuracy"], label="accuracy")
    plt.plot(r.history["val_accuracy"], label="val accuracy")
    plt.legend
    plt.show()

def main():
    # Start with a samll T and increase it later.
    T = 10
    # Declare the D
    D = 1
    #I first use the short distance pattern. Declare the X and Y and N
    X, Y, N = create_data(T, -1, -2, -3)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, shuffle=False)
    #Try a linear model first(Using short distance pattern) - note: it is classification now
    #Train the model with validation_split = 0.5
    X = np.expand_dims(X, -1)
    model = Linear()
    model.compile(optimizer=Adam(lr=1e-2),
                      loss=binary_crossentropy,
                      metrics=Accuracy)
    r = model.fit(X_train, Y_train, epochs=80, validation_data=(X_val, Y_val))
    #Plot the training results
    training_results(r)
    #Try a SimpleRNN model (Using short distance pattern)

    #Plot the training results

    #Try a LSTM model (Using short distance pattern)

    #Plot the training results

    #Try a GRU model (Using short distance pattern)

    #Plot the training results

    #Use a larger T value then compare the performances

if __name__ == '__main__':
    main()