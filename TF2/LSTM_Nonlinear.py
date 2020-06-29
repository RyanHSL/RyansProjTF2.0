'''
This script is comparing the Autogressive model with the SimpleRNN model and the LSTM model
Since the date is nonperidic, it is much harder for an AR model(as well as SimpleRNN and LSTM) to predict.
In this case, the SimpleRNN and LSTM can do better job than the AR model.
LSTM is better than SimpleRNN in finding the long term dependencies, but in this scenario, there is no long term dependency.
'''
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

#Make the original data sin(wx^2)
series = np.sin(0.1*np.arange(400)**2) + np.random.randn(400) * 0.1
#Plot the data
plt.plot(series)
plt.show()
###Build the dataset
#Declare N, T, D, M, K, X, Y
T, D, M , K = 10, 1, 15, 1
X = []
Y = []

for t in range(len(series) - T):
    x = series[t:t + T]
    y = series[t + T]
    X.append(x)
    Y.append(y)
#Reshape the input so that it fits the linear model
X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(Y)
print(X.shape, Y.shape)
###Try autoregressive linear model
i = Input(shape=X[0].shape)
x = Dense(1)(i)
model = Model(i, x)
#Train the RNN
model.compile(optimizer=Adam(lr=0.01),
              loss="mse",
              metrics=["accuracy"])

r = model.fit(X[:-N//2], Y[:-N//2], epochs=80, validation_data=(X[-N//2:], Y[-N//2:]))
#Plot the loss per iteration
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend
plt.show()
#Multi-step forecast
prediction_target = Y[-N//2:]
predictions = []
last_x = X[-N//2]

while len(predictions) < len(prediction_target):
    p = model.predict(last_x.reshape(1, -1))[0, 0]

    predictions.append(p)
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
#Plot the validation target and validation predictions
plt.plot(predictions, label="predictions")
plt.plot(prediction_target, label="targets")
plt.legend
plt.show()
###Try the SimpleRNN model
X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = len(Y)
print(X.shape, Y.shape)

i = Input(shape=X[0].shape)
x = SimpleRNN(M, activation="relu")(i)
x = Dense(1)(x)
model = Model(i, x)

model.compile(optimizer=Adam(lr=0.01),
              loss="mse",
              metrics=["accuracy"])

r = model.fit(X[:-N//2], Y[:-N//2], epochs=80, validation_data=(X[-N//2:], Y[-N//2:]))
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend
plt.show()

prediction_target = Y[-N//2:]
predictions = []
last_x = X[-N//2]

while len(predictions) < len(prediction_target):
    p = model.predict(last_x.reshape(1, -1, 1))[0, 0]

    predictions.append(p)
    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(predictions, label="predictions")
plt.plot(prediction_target, label="targets")
plt.legend
plt.show()

###Try the LSTM model
i = Input(shape=X[0].shape)
x = LSTM(M)(i) #Using default activation(tanh), since it maybe incompatible with Tensorflow GPU
x = Dense(1)(x)
model = Model(i, x)

model.compile(optimizer=Adam(lr=0.01),
              loss="mse",
              metrics=["accuracy"])

r = model.fit(X[:-N//2], Y[:-N//2], epochs=80, validation_data=(X[-N//2:], Y[-N//2:]))
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend
plt.show()

prediction_target = Y[-N//2:]
predictions = []
last_x = X[-N//2]

while len(predictions) < len(prediction_target):
    p = model.predict(last_x.reshape(1, -1, 1))[0, 0]

    predictions.append(p)
    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(predictions, label="predictions")
plt.plot(prediction_target, label="targets")
plt.legend
plt.show()