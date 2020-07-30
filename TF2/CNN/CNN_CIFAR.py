from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_model(inputs, K):
    i = inputs
    x = Conv2D(32, (3, 3), padding="same")(i)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = MaxPooling2D(2, 2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = MaxPooling2D(2, 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = MaxPooling2D(2, 2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(512, (3, 3), padding="same")(x)
    x = MaxPooling2D(2, 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), padding="same")(x)
    x = MaxPooling2D(2, 2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(K, activation="softmax")(x)
    model = Model(i, x)

    return model

def get_data():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train, X_test = X_train/np.float32(255), X_test/np.float32(255)
    Y_train, Y_test = Y_train.flatten(), Y_test.flatten()
    K = len(set(Y_train))

    return X_train, Y_train, X_test, Y_test, K

def main():
    X_train, Y_train, X_test, Y_test, K = get_data()
    print(X_train[0].shape)
    i = Input(shape=X_train[0].shape)
    batch_size = 128

    model = get_model(i, K)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    data_generator = ImageDataGenerator(rotation_range=45 ,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    train_generator = data_generator.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
    step_size = X_train.shape[0] // batch_size
    r = model.fit(train_generator, validation_data=(X_test, Y_test), steps_per_epoch=step_size, epochs=50)

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

    plt.plot(r.history["accuracy"], label="acc")
    plt.plot(r.history["val_accuracy"], label="val acc")
    plt.legend()
    plt.show()

    return



if __name__ == "__main__":
    main()