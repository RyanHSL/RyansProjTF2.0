from __future__ import print_function, division
from builtins import range

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
import matplotlib.pyplot as plt
tf1.disable_v2_behavior()

#create random training data
#define the number of samples for each class
Nclass = 500
#define the input features(dimensionality of input)
D = 2
#define the hidden layer size
M = 3
#define the number of classes
K = 3

#randomize the X1, X2, X3 and center them at [0, -2]. [2, 2], [-2, 2]. Stack them together to create a float32 X matrix
X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)
#Define Y as an empty array
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
#scatter plot the data
plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
plt.show()

#get the length of Y
N = len(Y)
#turn Y into an indicator matrix for training
indicator = np.zeros((N, K))
for i in range(N):
    indicator[i, Y[i]] = 1
#define an init_weights function to random normalize the variables. Standard deviation is 0.01
def init_weights(shape):
    return tf.random_normal_initializer(shape)
#define the forward function which returns the logits using sigmoid and matrix multiplication
def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2
#Create the place holders for X and Y. Type is float32 and the shape is [None, D] and [None, K]
tfX = tf1.placeholder(tf.int32)
tfY = tf1.placeholder(tf.int32)
#use init_weights function to initialize the W1, b1, W2, b2
W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights((K))
#use forward function to get the logits
pY_x = forward(X, W1, b1, W2, b2)
#calculate the cast using softmax cross entropy with logits
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tfY, logits = pY_x))
# WARNING: This op expects unscaled logits,
# since it performs a softmax on logits
# internally for efficiency.
# Do not call this op with the output of softmax,
# as it will produce incorrect results.

#define the train_op using tensorflow.train.GradientDescentOptimizer. Input parameter is the learning rate
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
#use tensorflow argmax to predict the logits output. Input parameter is the axis on which to choose the max
predict_op = tf.nn.argmax(pY_x, 1)
#initialize a tensorflow session
sess = tf.session()
#initialize all variables using tensorflow global_variables_initializer
init = tf.global_variable_initializer()
#run the initialization in tensorflow
sess.run(init)
#loop through 1000 epochs
for i in range(1000):
    #run the train_op in the session
    sess.run(train_op, feed_dict = {tfX: X, tfY: Y})
    #get the prediction by running predict_op in the session
    pred = sess.run(predict_op, feed_dict = {tfX: X, tfY: Y})
    #print the accuracy for every 100 element
    if i % 100 == 0:
        print("accuracy", np.mean(Y == pred))