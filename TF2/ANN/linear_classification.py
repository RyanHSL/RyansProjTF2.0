from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# data = load_breast_cancer()
# # type(data)
# # data.keys()
# # data.data.shape
# # data.target
# # data.target_names
# # data.target.shape
# # data.feature_names
#
# X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.33)
# N, D = X_train.shape
#
# scalar = StandardScaler()
# X_train = scalar.fit_transform(X_train)
# #X_test = np.reshape(-1, 1)
# X_test = scalar.transform(X_test) #do not expose the test data to the training pipe line
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(D,)),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
#
# model.compile(optimizer="adam",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
#
# r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100)
# print("Train Score: ", model.evaluate(X_train, Y_train))
# print("Test Score: ", model.evaluate(X_test, Y_test))
#
# plt.plot(r.history["loss"], label = "loss")
# plt.plot(r.history["val_loss"], label = "val_loss")
# plt.legend()
# plt.show()

#Save the model to a file

#Check that the model file exists

#Load the model and evaluate it and confirm that it still works

def get_data():
    data = load_breast_cancer()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)

    return x_train, y_train, x_test, y_test

class Classifier(tf.keras.models.Model):

    def __init__(self, input_shape):
        super(Classifier, self).__init__()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        self.model = model

        return

    def call(self, x):
        m = self.model(x)

        return m

def main():
    X_train, Y_train, X_test, Y_test = get_data()
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    model = Classifier(X_train[0].shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    """
    def schedule(epoch, lr):
        if epoch > 50:
            return lr * tf.math.exp(-0.1)
        return lr
        
    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
        
    r = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), batch_size=128, callbacks=[scheduler])
    """

    r = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), batch_size=128)

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    main()