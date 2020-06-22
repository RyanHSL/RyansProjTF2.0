import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

#define a y2indicator function to create a y indicator
# def y2indicator(y, K):
#     N = len(y)
#     ind = np.zeros((N, K))
#     for i in range(K): #LOOP THROUGH N ELEMENTS!!!!!!!!!!!!
#         ind[i, y[i]] = 1
#     return ind
def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
#get X, Y using get_data, shuffle them and change Y's element type to int32
X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)
#set the number of hidden unit M which is 5, input features number, and output number
M = 5
D = X.shape[1]
K = len(set(Y))
#seperate the data into test and train sets where Y are the last 100 elements and X are the rest. Create Y train and Y test indicators using y2indicator function
x_train = X[:-100]
y_train = Y[:-100]
x_test = X[-100:]
y_test = Y[-100:]
y_train_indicator = y2indicator(y_train, K)
y_test_indicator = y2indicator(y_test, K)
#randomize the W1, b1 and W2, b2
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)
#define the softmax function
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)
#define the forward function which takes input and all weights and biases and also returns hidden unit Z. Notice: Z is a tanh function
def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z
#define the predict function of P Y given X
def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis = 1)
#define the classification rate function
def classification_rate(T, pY):
    return np.mean(T == pY)
#define the cross entropy function which takes Target and pY
def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))
#create a list of train cost and test cost, set the learning rate
train_costs = []
test_costs = []
learning_rate = 0.001
#run through 10000 epochs. In each loop, calculate the py_train, py_test and Ztrain, Ztest using forward function,
#calculate train cost and test cost using cross_entropy function and append them to the train cost and test cost arrays
for i in range(10000):
    py_train, Ztrain = forward(x_train, W1, b1, W2, b2)
    py_test, Ztest = forward(x_test, W1, b1, W2, b2)

    train_cost = cross_entropy(y_train_indicator, py_train)
    test_cost = cross_entropy(y_test_indicator, py_test)
    train_costs.append(train_cost)
    test_costs.append(test_cost)
#do the gradient decent of W2, b2 and W1, b1 and calculate the derivative of Z
    # W2 -= learning_rate*Ztrain.T.dot(py_train - y_train_indicator)
    # b2 -= learning_rate*(py_train - y_train_indicator).sum(axis = 0)
    # dZ = (py_train - y_train_indicator).dot(W2.T)*(1 - Ztrain*Ztrain)
    # W1 -= learning_rate*x_train.T.dot(dZ)
    # b1 -= learning_rate*dZ.sum(axis = 0)
    W2 -= learning_rate*Ztrain.T.dot(py_train - y_train_indicator)
    b2 -= learning_rate*(py_train - y_train_indicator).sum(axis=0)
    dZ = (py_train - y_train_indicator).dot(W2.T) * (1 - Ztrain*Ztrain)
    W1 -= learning_rate*x_train.T.dot(dZ)
    b1 -= learning_rate*dZ.sum(axis=0)
#print train cost and test cost every 1000 epochs
    if i % 1000 == 0:
        print(i, train_cost, test_cost)
#print train costs and test costs
print("The training classification rate is ", classification_rate(y_train, predict(py_train)))
print("The test classification rate is ", classification_rate(y_test, predict(py_test)))
legend1 = plt.plot(train_costs, label = "train costs")
legend2 = plt.plot(test_costs, label = "test costs")
plt.legend([legend1, legend2])
plt.show()
# def y2indicator(y, K):
#     N = len(y)
#     ind = np.zeros((N, K))
#     for i in range(N):
#         ind[i, y[i]] = 1
#     return ind
#
# # x_train, y_train, x_test, y_test = get_data()
# X, Y = get_data()
# X, Y = shuffle(X, Y)
# Y = Y.astype(np.int32)
# x_train = X[:-100]
# y_train = Y[:-100]
# x_test = X[-100:]
# y_test = Y[-100:]
# D = x_train.shape[1]
# K = len(set(Y))
# M = 5 # num hidden units
#
# # convert to indicator
# y_train_ind = y2indicator(y_train, K)
# y_test_ind = y2indicator(y_test, K)
#
# # randomly initialize weights
# W1 = np.random.randn(D, M)
# b1 = np.zeros(M)
# W2 = np.random.randn(M, K)
# b2 = np.zeros(K)
#
# # make predictions
# def softmax(a):
#     expA = np.exp(a)
#     return expA / expA.sum(axis=1, keepdims=True)
#
# def forward(X, W1, b1, W2, b2):
#     Z = np.tanh(X.dot(W1) + b1)
#     return softmax(Z.dot(W2) + b2), Z
#
# def predict(P_Y_given_X):
#     return np.argmax(P_Y_given_X, axis=1)
#
# # calculate the accuracy
# def classification_rate(Y, P):
#     return np.mean(Y == P)
#
# def cross_entropy(T, pY):
#     return -np.mean(T*np.log(pY))
#
#
# # train loop
# train_costs = []
# test_costs = []
# learning_rate = 0.001
# for i in range(10000):
#     py_train, Ztrain = forward(x_train, W1, b1, W2, b2)
#     py_test, Ztest = forward(x_test, W1, b1, W2, b2)
#
#     ctrain = cross_entropy(y_train_ind, py_train)
#     ctest = cross_entropy(y_test_ind, py_test)
#     train_costs.append(ctrain)
#     test_costs.append(ctest)
#
#     # gradient descent
#     W2 -= learning_rate*Ztrain.T.dot(py_train - y_train_ind)
#     b2 -= learning_rate*(py_train - y_train_ind).sum(axis=0)
#     dZ = (py_train - y_train_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
#     W1 -= learning_rate*x_train.T.dot(dZ)
#     b1 -= learning_rate*dZ.sum(axis=0)
#     if i % 1000 == 0:
#         print(i, ctrain, ctest)
#
# print("Final train classification_rate:", classification_rate(y_train, predict(py_train)))
# print("Final test classification_rate:", classification_rate(y_test, predict(py_test)))
#
# legend1, = plt.plot(train_costs, label='train cost')
# legend2, = plt.plot(test_costs, label='test cost')
# plt.legend([legend1, legend2])
# plt.show()