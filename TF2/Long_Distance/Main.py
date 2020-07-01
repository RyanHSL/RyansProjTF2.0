# from Linear import Linear
# from SimpleRNN import SimpleRNN
# from LSTM import LSTM
# from GRU import GRU

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import  train_test_split
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, GlobalAvgPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid, relu

import tensorflow as tf
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
def create_data(SampleNum, T, i1, i2, i3):
    X, Y = [], []
    for t in range(SampleNum):
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

def Linear(optimizer, loss, X_train, Y_train, epochs, X_val, Y_val):
    i = Input(shape=X_train[0].shape)
    x = Dense(1, activation=sigmoid)(i)
    model = Model(i, x)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=["accuracy"])
    r = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))
    training_results(r)

def SimpleRNN_Model(activation, optimizer, loss, X_train, Y_train, epochs, X_val, Y_val):
    i = Input(shape=X_train[0].shape)
    x = SimpleRNN(15, activation=activation)(i)
    x = Dense(1, activation=sigmoid)(x)
    model = Model(i, x)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=["accuracy"])
    r = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))
    training_results(r)

def LSTM_Model(activation, optimizer, loss, X_train, Y_train, epochs, X_val, Y_val):
    i = Input(shape=X_train[0].shape)
    x = LSTM(15, activation=activation)(i)
    x = Dense(1, activation=sigmoid)(x)
    model = Model(i, x)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=["accuracy"])
    r = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))
    training_results(r)
    
def LSTM_Model_Enhanced(activation, optimizer, loss, X_train, Y_train, epochs, X_val, Y_val):
    i = Input(shape=X_train[0].shape)
    x = LSTM(15, activation=activation, return_sequences=True)(i)
    x = GlobalAvgPool1D()(x)
    x = Dense(1, activation=sigmoid)(x)
    model = Model(i, x)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=["accuracy"])
    r = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))
    training_results(r)

def GRU_Model(activation, optimizer, loss, X_train, Y_train, epochs, X_val, Y_val):
    i = Input(shape=X_train[0].shape)
    x = GRU(15, activation=activation)(i)
    x = Dense(1, activation=sigmoid)(x)
    model = Model(i, x)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=["accuracy"])
    r = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))
    training_results(r)


def main():
    # Start with a samll T and increase it later.
    T = 10
    # Declare the D
    D = 1
    #I first use the short distance pattern. Declare the X and Y and N
    X, Y, N = create_data(5000, T, 0, 1, 2)
    #Try a linear model first(Using short distance pattern) - note: it is classification now
    #Train the model with validation_split = 0.5
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, shuffle=False)
    # Linear(Adam(lr=1e-2), binary_crossentropy, X_train, Y_train, 80, X_val, Y_val)
    #Try a SimpleRNN model (Using short distance pattern)
    X = np.expand_dims(X, -1)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, shuffle=False)
    # SimpleRNN_Model(relu, Adam(lr=1e-2), binary_crossentropy, X_train, Y_train, 80, X_val, Y_val)
    #Try a LSTM model (Using short distance pattern)s
    # LSTM_Model(relu, Adam(lr=1e-2), binary_crossentropy, X_train, Y_train, 200, X_val, Y_val)
    # LSTM_Model_Enhanced(relu, Adam(lr=1e-2), binary_crossentropy, X_train, Y_train, 200, X_val, Y_val)
    #Try a GRU model (Using short distance pattern)
    GRU_Model(relu, Adam(lr=1e-2), binary_crossentropy, X_train, Y_train, 400, X_val, Y_val)
    #Use a larger T value then compare the performances

if __name__ == '__main__':
    main()