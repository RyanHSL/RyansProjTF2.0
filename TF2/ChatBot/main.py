import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import random
import json

# nltk.download()
with open("intents.json") as file:
    data = json.load(file)

words, labels, docs, docs_x, docs_y = [], [], [], [], []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training, output = [], []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

tf.compat.v1.reset_default_graph()
# training = np.expand_dims(training, -1)
i = Input(shape=training[0].shape)
x = Dense(10, activation="relu")(i)
x = Dense(10, activation="relu")(x)
x = Dense(len(output[0]), activation="sigmoid")(x)
model = Model(i, x)

model.compile(optimizer="adam",
              loss=tf.compat.v1.losses.sparse_softmax_cross_entropy,
              metrics=["accuracy"])

model.fit(training, output, nb_epoch=1000, batch_size=8)
# model.save("model.h5")

