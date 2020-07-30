from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Regressor(tf.keras.models.Model):

    def __init__(self, input_shape):
        super(Regressor, self).__init__()

        model = Sequential([layers.Input(shape=input_shape),
                           # layers.Dense(32, activation="relu"),
                           layers.Dense(1)])
        self.model = model

        return

    def call(self, x):
        x = self.model(x)

        return x

class RNN(tf.keras.models.Model):

    def __init__(self, input_shape):
        super(RNN, self).__init__()

        i = layers.Input(shape=input_shape)
        x = layers.GRU(32, activation="relu")(i)
        x = layers.Dense(1)(x)

        self.model = tf.keras.models.Model(i, x)

        return

    def call(self, x):
        m = self.model(x)

        return m

def get_data():
    df = pd.read_csv("Moore.csv").values
    X, Y = df[:, 0].reshape(-1, 1), np.log(df[:, 1])
    X = X - X.mean()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    return  X_train, Y_train, X_test, Y_test

def get_RNN_data():
    df = pd.read_csv("Moore.csv").values
    series = df[:, 1]
    series = series - series.mean()
    N = len(series)
    T = 10
    X, Y = [], []

    for i in range(N - T):
        x = series[i:i + T]
        y = series[i + T]
        X.append(x)
        Y.append(y)

    X = np.array(X).reshape(-1, T, 1).astype(np.float32)
    Y = np.array(Y).astype(np.float32)
    plt.plot(series)
    plt.show()

    return X, Y, N

def main():
    X_train, Y_train, X_test, Y_test = get_data()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    plt.scatter(X_train, Y_train)
    plt.show()

    model = Regressor(X_train[0].shape)
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                  loss="mse")

    def schedule(epoch, lr):
        if epoch >= 50:
            return 1e-3
        return 1e-2

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    r = model.fit(X_train, Y_train, epochs=100, batch_size=128, callbacks=[scheduler], validation_data=(X_test, Y_test))

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.show()

    plt.scatter(X_test, Y_test)
    plt.show()
    prediction = model.predict(X_test)
    plt.scatter(X_test, prediction)
    plt.show()
    """
    RNN Approach which is not practical
    The values are too large so the losses also become very large
    """
    # X, Y, N = get_RNN_data()
    # X_train, X_test = X[:-N // 2], X[-N // 2:]
    # Y_train, Y_test = Y[:-N // 2], X[-N // 2:]
    #
    # model = RNN(X_train[0].shape)
    # model.compile(optimizer=Adam(lr=1e-2),
    #               loss="mse",
    #               metrics=["accuracy"])
    # r = model.fit(X_train, Y_train, epochs=100, batch_size=128)
    # plt.plot(r.history["loss"], label="loss")
    # # plt.plot(r.history["val_loss"], label="val loss")
    # plt.show()
    #
    # prediction_target = Y[-N//2:]
    # prediction = []
    # last_x = X[-N//2]
    #
    # while len(prediction) < len(prediction_target):
    #     p = model.predict(last_x.reshape(1, -1, 1))[0, 0]
    #     prediction.append(p)
    #
    #     last_x = np.roll(last_x, -1)
    #     last_x[-1] = p
    #
    # plt.plot(prediction, label="prediction")
    # plt.plot(prediction_target, label="target")
    # plt.legend()
    # plt.show()

    return

if __name__ == '__main__':
    main()