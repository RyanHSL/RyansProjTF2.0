from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
# print(df.head())
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
# print(df.head())
df.columns = ["labels", "data"]
# print(df.head())
df["b_labels"] = df["labels"].map({"ham": 0, "spam": 1})
print(df.head())

Y = df["b_labels"].values
data_train, data_test, Y_train, Y_test = train_test_split(df["data"], Y, test_size=0.33)

MAX_VOCAB_NUM = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_NUM, oov_token="<OOV>")
tokenizer.fit_on_texts(data_train)
train_sequences = tokenizer.texts_to_sequences(data_train)
test_sequences = tokenizer.texts_to_sequences(data_test)

word2Index = tokenizer.word_index
V = len(word2Index)
print("Found %s unique tokens" %V)

train_data = pad_sequences(train_sequences)
print("shape of training data: ", train_data.shape)
T = train_data.shape[1]

test_data = pad_sequences(test_sequences, maxlen=T)
print("shape of testing data: ", test_data.shape)

D = 20
M = 15

i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = Conv1D(32, 3, activation="relu")(x)
x = MaxPooling1D(2)(x)
x = Conv1D(64, 3, activation="relu")(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)
# x = GlobalMaxPooling1D()(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(i, x)

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
r = model.fit(train_data, Y_train, epochs=10, validation_data=(test_data, Y_test))

#Plot the loss per iteration
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend()
plt.show()
#Plot the accuracy per iteration
plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="val accuracy")
plt.legend()
plt.show()