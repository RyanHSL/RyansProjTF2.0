from tensorflow.keras.layers import LSTM, Input, Dense, Flatten, GlobalMaxPooling2D
from tensorflow.keras.models import Model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dataset = tf.keras.datasets.fashion_mnist

(X_train, Y_train), (X_test, Y_test) = dataset.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

i = Input(shape=X_train[0].shape)
x = LSTM(128)(i)
x = Dense(10, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

r = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend
plt.show()

plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="val accuracy")
plt.legend
plt.show()