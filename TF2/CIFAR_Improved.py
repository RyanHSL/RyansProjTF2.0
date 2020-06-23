from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Dense, GlobalMaxPool2D, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools

cifar10 = tf.keras.datasets.cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
Y_train, Y_test = Y_train.flatten(), Y_test.flatten()
print("X shape is: ", X_train.shape)
print("Y shape is: ", Y_train.shape)

K = len(set(Y_train))
print("The number of output classes is :", K)
#Build the model using functional API
#Add BatchNormalization after each Conv2D layer
#Use MaxPooling2D instead of strides = 2, since I am processing small image data
#Add Dropout before each Conv2D and Dense later
i = Input(shape=X_train[0].shape)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(i)
x = BatchNormalization()(x)
#x = MaxPooling2D(2, 2)(x)
#x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Dropout(0.2)(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
#x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Dropout(0.2)(x)

x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
#x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)

x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
#x = Dropout(0.2)(x)
x = Dense(K, activation="softmax")(x)

model = Model(i, x)
#Compile and fit
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50)
#Fit with data augmentation
#Note: If I run this after calling the previous model.fit(), it will continue training where it left off(Using the same weights and biases)
#Define the batch size
batch_size = 32
#Use ImageGenerator to create a data generator width_shift_range and height_shift_ranges are 0.1, horizaontal flips
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
#Use data generator flow(X_train, Y_train, batch_size) to get the train_generator
train_generator = data_generator.flow(X_train, Y_train, batch_size)
#Step per epoch is the shape of the first training data
steps_per_epoch = X_train.shape[0] // batch_size
#Fit the train generator
r = model.fit(train_generator, validation_data=(X_test, Y_test), steps_per_epoch=steps_per_epoch, epochs=50)

plt.plot(r.history["loss"], label="Loss")
plt.plot(r.history["val_loss"], label="Val Loss")
plt.legend
plt.show()

plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="Val accuracy")
plt.legend
plt.show()