from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

import os
import numpy as np
import matplotlib.pyplot as plt

max_vocab = 10000
max_review_length = 80

def get_data():
    np.random.seed(7)

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_vocab)
    x_train = pad_sequences(x_train, maxlen=max_review_length)
    x_test = pad_sequences(x_test, maxlen=max_review_length)

    return x_train, y_train, x_test, y_test

def plot(r):
    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

    plt.plot(r.history["accuracy"], label="accuracy")
    plt.plot(r.history["val_accuracy"], label="val accuracy")
    plt.legend()
    plt.show()

    return

class RNN(keras.Model):

    def __init__(self, units, num_classes, num_layers):
        super(RNN, self).__init__()
        self.rnn = keras.layers.LSTM(units, return_sequences=True)
        self.rnn2 = keras.layers.LSTM(units)
        self.embedding = keras.layers.Embedding(max_vocab, 100, input_length=max_review_length)
        self.fc = keras.layers.Dense(1)

        return

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.rnn2(x)
        x = self.fc(x)

        return x


def main():
    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 20

    x_train, y_train, x_test, y_test = get_data()

    model = RNN(units, num_classes, num_layers=2)


    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("Final test loss and accuracy:", scores)

    plot(r)

    return

if __name__ == "__main__":
    main()