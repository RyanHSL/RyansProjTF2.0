'''
In this scenario, SimpleRNN doesn't perform as well as autoregression model because it is too flexible,
but the flexibility of an RNN allows it to do more power things than a AR model at some point.
The prediction of this SimpleRNN model is quite inconsistent
'''

from tensorflow.keras.layers import Input, Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Make the original data(a sine wave)
series = np.sin(0.1 * np.arange(200)) + np.random.randn(200) * 0.1
#Plot it
plt.plot(series)
plt.show()
#Build the dataset
T = 10
X = []
Y = []

for t in range(len(series) - T):
    x = series[t: t + T]
    y = series[t + T]
    X.append(x)
    Y.append(y)
#Reshape X to a (N x T x D) matrix
X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = len(Y)
#Print the shape of X and Y
print(X.shape, Y.shape)
###Try autoregressive RNN model using Simple RNN
i = Input(shape=X[0].shape)
x = SimpleRNN(15, activation=tf.nn.relu)(i) #Activation for SimpleRNN is tanh by default
x = Dense(1)(x)
model = Model(i, x)

model.compile(optimizer=Adam(lr=0.01),
              loss=mse,
              metrics=["accuracy"])
#Train the RNN
r = model.fit(X[: -N//2], Y[: -N//2], epochs=80, validation_data=(X[-N//2 :], Y[-N // 2:]))
#Plot loss per iteration
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.show()
#Forcase future values(use only self-predictions for making future predictions)
validation_target = Y[-N // 2:]
validation_predictions = []
#declare the first validation input. It is a 1-D array of length T
last_x = X[-N // 2]
#Create a loop to do the prediction
while len(validation_predictions) < len(validation_target):
    #update the prediction list
    p = model.predict(last_x.reshape(1, T, 1))[0,0] #1x1 array -> scalar
    #Make the new validation input
    validation_predictions.append(p)
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
#Plot the targets and the predictions
plt.plot(validation_target, label="validation target")
plt.plot(validation_predictions, label="validation predictions")
plt.legend()
plt.show()