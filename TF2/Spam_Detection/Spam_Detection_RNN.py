from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, GlobalMaxPool1D, Dense, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the csv file 'spam.csv'
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
#Display the head of data
# print(df.head())
#Drop the unnecessary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
#Display the head of data
# print(df.head())
#Rename the columns to something better
df.columns = ["labels", "data"]
#Display the head of data
# print(df.head())
#Create binary labels
df["b_labels"] = df["labels"].map({"ham":0, "spam":1})
Y = df["b_labels"].values
print(df.head())
#Split up the data
data_train, data_test, Y_train, Y_test = train_test_split(df["data"], Y, test_size=0.33)
#Convert sentences to sequences
maxVocab = 20000
tokenizer = Tokenizer(num_words = maxVocab)
tokenizer.fit_on_texts(data_train)
train_sequences = tokenizer.texts_to_sequences(data_train)
test_sequences = tokenizer.texts_to_sequences(data_test)
#get word -> integer mapping and calculate and print the length of word index
word2Index = tokenizer.word_index
V = len(word2Index)
print("Find %s unique tokens" %V)
#Pad sequences so that we get a NxT matrix then print the shape of training data
training_data = pad_sequences(train_sequences)
print("Shape of training data: ", training_data.shape)
#Get the sequence length
T = training_data.shape[1]
#Pad sequences using maxlen T so that we get the test data then print the shape of test tensor
test_data = pad_sequences(test_sequences, maxlen=T)
print("Shape of test data: ", test_data.shape)
#Create the model
#Choose the embedding demensionality which is D
D = 20
#Define the hidden state dimensionality
M = 15
# Note: we actually want to the size of the embedding to (V + 1) x D,
# because the first index starts from 1 and not 0.
# Thus, if the final index of the embedding matrix is V,
# then it actually must have size V + 1.
i = Input(shape=(T, ))
x = Embedding(V + 1, D)(i)
x = LSTM(M, activation="relu", return_sequences=True)(x)
# x = GlobalMaxPool1D()(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(i, x)
#Compile and fit
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

r = model.fit(training_data, Y_train, epochs=10, validation_data=(test_data, Y_test))
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
