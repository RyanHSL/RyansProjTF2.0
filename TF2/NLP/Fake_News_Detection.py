from tensorflow.keras.layers import Embedding, LSTM, Input, Dense, Flatten, GlobalAveragePooling1D, SimpleRNN, GRU, Conv1D, Dropout, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data():
    df_fake = pd.read_csv("Data/news/Fake.csv")
    df_true = pd.read_csv("Data/news/True.csv")
    # Use the subject as data only, since the content has 116384 word index
    df_fake = df_fake.drop(["date", "subject", "text"], axis=1)
    df_true = df_true.drop(["date", "subject", "text"], axis=1)
    df_fake.columns = ["data"]
    df_true.columns = ["data"]
    df_fake["labels"] = [0 for _ in range(len(df_fake))]
    df_true["labels"] = [1 for _ in range(len(df_true))]
    df = df_fake.append(df_true, ignore_index=True)
    # df = df.sample(frac=1).reset_index(drop=True)
    # print(df.head())
    x, y = df["data"].values, df["labels"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)

    return x_train, y_train, x_test, y_test

def text_to_sequence(train_text, test_text):
    maxVocab = 20000
    tokenizer = Tokenizer(num_words=maxVocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)
    train_sequence = tokenizer.texts_to_sequences(train_text)
    test_sequence = tokenizer.texts_to_sequences(test_text)

    word2vec = tokenizer.word_index
    v = len(word2vec) + 1 # The word index starts from 1. 0 is padding

    train_data = pad_sequences(train_sequence) # NxT matrix
    t = train_data.shape[1]
    test_data = pad_sequences(test_sequence, maxlen=t)

    return train_data, test_data, v, t

class LSTMModel(Model):
    def __init__(self, input_shape, M, D, V):
        super(LSTMModel, self).__init__()
        self.M = M
        self.D = D # Embedding dimension
        self.V = V

        i = Input(shape=input_shape)
        x = Embedding(self.V, self.D)(i)
        x = LSTM(self.M, activation="relu", return_sequences=True)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(i, x)
        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

class GRUModel(Model):
    def __init__(self, input_shape, M, D, V):
        super(GRUModel, self).__init__()
        self.M = M
        self.D = D
        self.V = V

        i = Input(shape=input_shape)
        x = Embedding(self.V, self.D)(i)
        x = GRU(self.M, activation=tf.nn.relu)(x)
        x = Dense(1, activation=tf.nn.sigmoid)(x)
        model = Model(i, x)

        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

class SimpleRNNModel(Model):
    def __init__(self, input_shape, M, D, V):
        super(SimpleRNNModel, self).__init__()
        self.M = M
        self.D = D
        self.V = V

        i = Input(shape=input_shape)
        x = Embedding(self.V, self.D)(i)
        x = SimpleRNN(self.M, activation=tf.nn.relu)(x)
        x = Dense(1, activation=tf.nn.sigmoid)(x)
        model = Model(i, x)

        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

class CNNModel(Model):
    def __init__(self, input_shape, M, D, V):
        super(CNNModel, self).__init__()
        self.M = M
        self.D = D
        self.V = V

        i = Input(shape=input_shape)
        x = Embedding(self.V, self.D)(i)
        x = Conv1D(self.M, 3, padding="same", activation=tf.nn.relu)(x)
        x = MaxPooling1D(2, 2)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(self.M * 2, 3, padding="same", activation=tf.nn.relu)(x)
        x = MaxPooling1D(2, 2)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        # x = Dense(32, activation=tf.nn.relu)(x)
        x = Dense(1, activation=tf.nn.sigmoid)(x)
        model = Model(i, x)

        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

def main():
    X_train, Y_train, X_test, Y_test = get_data()
    X_train, X_test, V, T = text_to_sequence(X_train, X_test)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # rnnModel = LSTMModel(X_train[0].shape, 15, 20, V) # loss: 0.0247 - accuracy: 0.9927 - val_loss: 0.1419 - val_accuracy: 0.9561
        # rnnModel = SimpleRNNModel(X_train[0].shape, 15, 20, V) # loss: 0.0032 - accuracy: 0.9992 - val_loss: 0.1444 - val_accuracy: 0.9665
        # rnnModel = GRUModel(X_train[0].shape, 15, 20, V) # loss: 0.0032 - accuracy: 0.9990 - val_loss: 0.1462 - val_accuracy: 0.9677
        rnnModel = CNNModel(X_train[0].shape, 32, 20, V) # loss: 0.0017 - accuracy: 0.9996 - val_loss: 0.1312 - val_accuracy: 0.9721
        rnnModel.compile(optimizer=Adam(lr=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"])

    r = rnnModel.fit(X_train, Y_train, epochs=8, batch_size=128, validation_data=(X_test, Y_test))

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()
    plt.plot(r.history["accuracy"], label="accuracy")
    plt.plot(r.history["val_accuracy"], label="val accuracy")
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()