from tensorflow.keras.layers import Input, Dense, LSTM, SimpleRNN, GRU, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import requests

class LSTMModel(Model):
    def __init__(self, input_shape, V, sequence_length, M=100, D=50):
        super(LSTMModel, self).__init__()
        self.M = M
        self.D = D # Embedding dimension
        self.V = V
        self.sequence_length = sequence_length

        i = Input(shape=input_shape)
        x = Embedding(self.V, self.D, input_length=self.sequence_length)(i)
        x = LSTM(self.M, activation=tf.nn.relu, return_sequences=True)(x)
        x = LSTM(self.M, activation=tf.nn.relu, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(self.M, activation=tf.nn.relu)(x)
        x = Dense(self.V, activation=tf.nn.softmax)(x) # Note: the output length is V
        model = Model(i, x)
        model.summary()
        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

def GRUModel(Model):
    def __init__(self, input_shape, V):
        super(GRUModel, self).__init__()
        self.V = V
        self.M = 15
        self.D = 20
        model = Sequential([Input(shape=input_shape),
                            Embedding(self.V, self.D),
                            GRU(self.M, activation=tf.nn.relu),
                            Dense(self.V, activation=tf.nn.softmax)])
        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

def clean_text(doc):
    tokens = doc.split() # Seperated by white spaces
    table = str.maketrans("", "", string.punctuation) # Remove punctuation from any string
    tokens = [w.translate(table) for w in tokens]
    tokens = [word.lower() for word in tokens if word.isalpha()]

    return tokens

def get_data():
    # r = requests.get("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt")
    #
    # if r.status_code == 200:
    #     data = r.text.split("\n")
    #     data = data[253:] # The poem starts from line 253
    #     # print(f"the first line: {data[0]}\nthe total line number: {len(data)}")
    #     data = " ".join(data)
    #
    #     tokens = clean_text(data)
    #     print(tokens[:50])
    #     print(len(set(tokens)), "unique words") # 27956 unique words
    #     maxVocab = 50000
    #     tokenizer = Tokenizer(num_words=maxVocab)
    #     tokenizer.fit_on_texts(tokens)
    #     word_sequences = tokenizer.texts_to_sequences(tokens)
    #     word2vec = tokenizer.word_index
    #     v = len(word2vec) + 1
    #     sequences = pad_sequences(word_sequences)

    df = pd.read_csv("Data/Trump_Tweets/trumptweets.csv")
    df.drop(["id", "link", "date", "retweets", "favorites", "mentions", "hashtags", "geo"], axis=1)
    # data = df["content"].values
    df['content'] = df['content'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True).replace(r'https\S+', '', regex=True)
    data = df["content"].values
    data = data[:2000]
    # data = " ".join(data)
    # tokens = clean_text(data)
    print(data[:50])
    print(len(set(data)), "unique words")
    maxVocab = 20000
    tokenizer = Tokenizer(num_words=maxVocab)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    word2vec = tokenizer.word_index
    v = len(word2vec) + 1
    # sequences = pad_sequences(sequences)

    return tokenizer, sequences, v

def split_time_series_data(doc, T):
    X, Y = [], []

    for t in range(len(doc) - T):
        x = doc[t:t + T]
        y = doc[t + T]
        X.append(x)
        Y.append(y)

    X = np.array(X).reshape(-1, T)
    Y = np.array(Y)
    N = len(Y)

    X_train, X_test = X[:-N//2], X[-N//2:]
    Y_train, Y_test = Y[:-N//2], Y[-N//2:]

    return X_train, X_test, Y_train, Y_test, N

def split_data(data, V):
    datalist = []
    for d in data:
        if len(d) > 1:
            for i in range(2, len(d)):
                datalist.append(d[:i])
    # X_train, Y_train = data[:-N//2, :-1], data[:-N//2, -1]
    # X_test, Y_test = data[-N//2:, :-1]. data[-N//2:, -1]
    max_length = 20 # "NLP" (and any subsequent words) was ignored because we limit queries to 32 words.
    sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')

    X = sequences[:, :-1]
    Y = sequences[:, -1]
    Y = to_categorical(Y, num_classes=V)

    return X, Y


def main():
    tokenizer, sequences, V = get_data()
    # T = 50
    X, Y = split_data(sequences, V)
    model = LSTMModel(X[0].shape, V, X.shape[1])

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    # def schedule(epoch, lr):
    #     if epoch > 50:
    #         return 1e-3
    #     return 1e-2
    #
    # scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    # r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, callbacks=[scheduler], batch_size=1024)
    # r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=1024)
    r = model.fit(X, Y, epochs=200, batch_size = 256)

    plt.plot(r.history["loss"], label="loss")
    # plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

    # target = Y_test
    # predictions = []
    # last_x = X_test[0]
    #
    # while len(target) > len(predictions):
    #     p = model.predict(last_x.reshape(1, -1, 1))[0, 0]
    #
    #     predictions.append(p)
    #     last_x = np.roll(last_x, -1)
    #     last_x[-1] = p
    #
    # plt.plot(Y_test, label="target")
    # plt.plot(predictions, label="predictions")
    # plt.legend()
    # plt.show()

    text_seed = "Make America Great Again"
    tweet_length = 10
    for i in range(10):
        text = []
        for _ in range(tweet_length):
            encoded = tokenizer.texts_to_sequences([text_seed])
            encoded = pad_sequences(encoded, maxlen=X.shape[1], padding='pre')

            y_pred = np.argmax(model.predict(encoded), axis=-1)

            predicted_word = ""
            for word, index in tokenizer.word_index.items():
                if index == y_pred:
                    predicted_word = word
                    break

            text_seed = text_seed + ' ' + predicted_word
            text.append(predicted_word)

        text_seed = text[-1]
        text = ' '.join(text)
        print(text)

    model.save_weights("trump")
    return

if __name__ == "__main__":
    main()