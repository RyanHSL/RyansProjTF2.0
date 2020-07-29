'''
This script shows how SimpleRNN is working.
'''
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.models import Model

import numpy as np

#Create the data: N, T, D, K, X
N, T, D, K = 1, 10, 3, 2
X = np.random.randn(N, T, D)
#Create the RNN model and M for the hidden unit size
M = 5
i = Input(shape=(T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)

model = Model(i, x)
#Since the data is random, I do not train the model
#Predict the output and print it
prediction = model.predict(X)
print("The SimpleRNN prediction is: ", prediction)
#Get the model summary
model.summary()
#Get the weights of the SimpleRNN layer
Wx, Wh, bh = model.layers[1].get_weights()
#Print the shape of each weights in the SimpleRNN layer
print(Wx.shape, Wh.shape, bh.shape)
#Get the weights of the output layer
Wo, bo = model.layers[2].get_weights()
#Replicate the output using activation function and linear model
#Initialize the hidden state
last_h = np.zeros(M)
#Declare the one and only sample
x = X[0]
#Create an empty prediction list
predictions = []
#Loop through all T series and calculatete the output
for t in range(T):
    #Use tanh as the activation function
    h = np.tanh(x[t].dot(Wx) + last_h.dot(Wh) + bh)
    #I only care about this value on the last iteration
    p = h.dot(Wo) + bo
    #Append the prediction to the prediction list
    predictions.append(p)
    last_h = h
#Print the final output
print("The tanh + linear prediction is: ", predictions[-1])
#Calculatet the output for multiplet samples at once (N > 1)
N, T, D, K = 5, 10, 3, 2
X = np.random.randn(N, T, D)
M = 5
i = Input(shape=(T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)

model = Model(i, x)
prediction = model.predict(X)
print("The SimpleRNN prediction is: ", prediction)
Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()

for n in range(N):
    x = X[n]
    last_h = np.zeros(M)
    predictions = []
    for t in range(T):
        #Use tanh as the activation function
        h = np.tanh(x[t].dot(Wx) + last_h.dot(Wh) + bh)
        #I only care about this value on the last iteration
        p = h.dot(Wo) + bo
        #Append the prediction to the prediction list
        predictions.append(p)
        last_h = h

    print("The tanh + linear prediction is: ", predictions[-1])
