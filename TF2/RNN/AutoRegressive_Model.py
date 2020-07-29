from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Make the original data, which is a 200 samples SINE wave with noise
series = np.sin(0.1 * np.arange(200)) + np.random.randn(200)*0.1
#Plot the data
plt.plot(series)
plt.show()
###Build the dataset: T, X, Y. X and Y are lists
#Convert X and Y to numpy arrays then reshape X to a NxT array.
#Print the X shape and Y shape
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x = series[t: t + T]
    y = series[t + T]
    X.append(x)
    Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print(X.shape, Y.shape)
###Try autoregressive linear model
#Build a feedforward ANN using functional API then compile it using mean squared error loss and 0.1 learning rate
i = Input(shape=(T,))
x = Dense(1)(i)

model = Model(i, x)
model.compile(optimizer=Adam(lr=0.1),
              loss="mse",
              metrics=["accuracy"])
#Train the RNN.
#Note: Since it is predicting the future value, the training data and test data cannot be randomly distributed.
#They need to be seperated half by half. Seperate them using floor divide
X_train, X_test = X[:-N // 2], X[-N // 2:]
Y_train, Y_test = Y[:-N // 2], Y[-N // 2:]

r = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))
#Plot loss per iteration
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend
plt.show()
#Forecast future values(use only self-predictions for making future predictions)
#Declare the validation target and validation prediction list
validation_target = Y[-N // 2:]
validation_predictions = []
#Declare the first validation input. It is an 1-D array of length T
last_x = X[-N // 2]
#While the validation length is less than validation target length, predict the p
#Note: reshape the x and p so that they are a 1x1 arrays
#Update the predictions list by appending the new prediction
#Make the new input by shifting the first validation input one index to the left and add the prediction to the end
while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, -1))[0, 0]

    validation_predictions.append(p)
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
#Plot the validation_target and validation_predictions
plt.plot(validation_target, label="validation target")
plt.plot(validation_predictions, label="validation predictions")
plt.legend
plt.show()