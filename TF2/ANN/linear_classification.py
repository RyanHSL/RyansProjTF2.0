from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()
# type(data)
# data.keys()
# data.data.shape
# data.target
# data.target_names
# data.target.shape
# data.feature_names

X_train, Y_train, X_test, Y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
#X_test = np.reshape(-1, 1)
X_test = scalar.transform(X_test) #do not expose the test data to the training pipe line

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuray"])

r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100)
print("Train Score: ", model.evaluate(X_train, Y_train))
print("Test Score: ", model.evaluate(X_test, Y_test))

plt.plot(r.history["loss"], label = "loss")
plt.plot(r.history["val_loss"], label = "val_loss")
plt.legend()
plt.show()

#Save the model to a file

#Check that the model file exists

#Load the model and evaluate it and confirm that it still works