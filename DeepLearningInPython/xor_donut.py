# import numpy as np
# import matplotlib.pyplot as plt
#
# #define a forward function to do the binary classification using sigmoid to do activation. Return Y and Z
# def forward(x, W1, b1, W2, b2):
#     Z  = 1/(1 - np.exp(-x)) #sigmoid
#     activation = Z.dot(W1) + b1
#     Y = 1/(1 + np.exp(-activation)) #tanh
#     return Y, Z
# #define the predict function to round up the Y
# def predict(x, W1, b1, W2, b2):
#     Y, _ = forward(x, W1, b1, W2, b2)
#     return np.round(Y)
# #define the derivative functions of W1, b1, W2, b2. The last hidden layer is a sigmoid function and the first hidden layer is a tanh function
# def derivative_w2(Z, T, Y):
#     return (Y - T).dot(Z)
# def derivative_b2(T, Y):
#     return (Y - T).sum()
# def derivative_w1(X, Z, T, Y, W2):
#     dZ = np.outer(T - Y, W2)*(1 - Z*Z)
#     return X.T.dot(dZ)
# def derivative_b1(Z, T, Y, W2):
#     dZ = np.outer(T - Y, W2)*(1 - Z*Z)
#     return dZ.sum(axis = 0)
# #define a cost function to do the binary_cross_entropy
# def cost(T, Y):
#     tot = 0
#     for i in range(len(T)):
#         if T[i] == 1:
#             tot += np.log(Y[i])
#         else:
#             tot += np.log(1 - Y[i])
#     return tot
# #define a test XOR function
# def test_XOR():
#     #create the input matrix X and the output matrix Y
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     Y = np.array([0, 1, 1, 0])
#     #randomize the W1, b1, W2, b2
#     W1 = np.random.randn(2, 4)
#     b1 = np.random.randn(4)
#     W2 = np.random.rand(4, 4)
#     b2 = np.random.rand(1)
#     #create a log likelihood array
#     LL = []
#     #define the learning rate
#     learning_rate = 0.0005
#     #define the regularization value
#     regularization = 0.
#     #intialize the last error rate
#     last_error_rate = None
#     #loop through 100000 epochs.
#     for i in range(100000):
#         #Use forward function to get pY and Z.
#         pY, Z = forward(X, W1, b1, W2, b2)
#         #Use cost function to calculate the log likelihood.
#         ll = cost(Y, pY)
#         #Use predict function to calculate the prediction
#         prediction = predict(X, W1, b1, W2, b2)
#         #Use the mean of absolute difference of prediction and Y to calculate the error rate
#         er = np.abs(prediction - Y).mean()
#         #Update the last error rate with new error rate. Print error rate, the Y value and the prediction
#         if er != last_error_rate:
#             last_error_rate = er
#             print("Error rate: ", er, " Y: ", Y, " Prediction: ", prediction)
#         #break the loop if the new log likelihood exists and decreases
#         if LL and ll < LL[-1]:
#             print("Early exit")
#             break
#         #append the log likelihood to the log likelihood array
#         LL.append(ll)
#         #do the gradient descent of W1, b1, W2, b2
#         W1 -= learning_rate*(derivative_w1(X, Z, Y, pY, W2) - regularization*W1)
#         b1 -= learning_rate*(derivative_b1(Z, Y, pY, W2) - regularization*b1)
#         W2 -= learning_rate*(derivative_w2(Z, Y, pY) - regularization*W2)
#         b2 -= learning_rate*(derivative_b2(Y, pY) - regularization*b2)
#     #print the classification rate which is 1 minus the mean of the absolute difference between Y and prediction
#     print("Classification rate: ", 1 - np.abs(Y - prediction).mean())
#
# #define a test_donut function
# def test_donout():
#     #define the number of samples, inner radius and outer radius
#     N = 1000
#     R_inner = 5
#     R_outer = 10
#     #distance from origin is radius + random normal
#     #angle theta is uniformly distributed between (0, 2pi)
#     R1 = np.random.randn(N//2) + R_inner
#     theta = 2*np.pi*np.random.random(N//2)
#     X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
#
#     R2 = np.random.randn(N//2) + R_outer
#     theta = 2*np.pi*np.random.random(N//2)
#     X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
#
#     X = np.concatenate([X_inner, X_outer])
#     Y = np.array([0]*(N//2) + [1]*(N//2))
#
#     #define the number of hidden units
#     n_hidden = 8
#     #randomize W1, b1, W2, b2
#     W1 = np.random.randn(2, n_hidden)
#     b1 = np.random.randn(n_hidden)
#     W2 = np.random.randn(n_hidden)
#     b2 = np.random.randn(1)
#     #create a log likelihood array
#     LL = []
#     #define learning rate, regularization, last error rate
#     learning_rate = 0.0005
#     regularization = 0.
#     last_error_rate = None
#     #loop through 160000 epochs
#     for i in range(160000):
#         #use forward function to get pY and Z
#         pY, Z = forward(X, W1, b1, W2, b2)
#         #use cost function to calculate the log likelihood
#         ll = cost(Y, pY)
#         #use predict function to get the prediction
#         prediction = predict(X, W1, b1, W2, b2)
#         #error rate is the mean of the absolute difference between prediction and Y
#         er = np.abs(prediction - Y).mean()
#         #append the log likelihood
#         LL.append(ll)
#         #do the gradient descent of W1, b1, W2, b2 Notice: the gradient descent equals learning rate times the derivative of W/b minus regularization times W/b
#         W1 -= learning_rate*(derivative_w1(X, Z, Y, pY, W2) - regularization*W1)
#         b1 -= learning_rate*(derivative_b1(Z, Y, pY, W2) - regularization*b1)
#         W2 -= learning_rate*(derivative_w2(Z, Y, pY) - regularization*W2)
#         b2 -= learning_rate*(derivative_b2(Y, pY) - regularization*b2)
#         #for every 100 element, print the log likelihood and classification rate, which is 1 minus error rate
#         if i%100 == 0:
#             print("log likelihood: ", ll, " classification rate: ", 1 - er)
#     #plot the log likelihood array
#     plt.plot(LL)
#     plt.show()
#
# if __name__ == '__main__':
#     test_XOR()
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

# for binary classification! no softmax here

def forward(X, W1, b1, W2, b2):
    # sigmoid
    # Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))

    # tanh
    # Z = np.tanh(X.dot(W1) + b1)

    # relu
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)

    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z


def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)


def derivative_w2(Z, T, Y):
    # Z is (N, M)
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    return (T - Y).sum()


def derivative_w1(X, Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return X.T.dot(dZ)


def derivative_b1(Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return dZ.sum(axis=0)


def get_log_likelihood(T, Y):
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))



def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 5)
    b1 = np.zeros(5)
    W2 = np.random.randn(5)
    b2 = 0
    LL = [] # keep track of log-likelihoods
    learning_rate = 1e-2
    regularization = 0.
    last_error_rate = None
    for i in range(30000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = get_log_likelihood(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.mean(prediction != Y)

        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 1000 == 0:
            print(ll)

    print("final classification rate:", np.mean(prediction == Y))
    plt.plot(LL)
    plt.show()


def test_donut():
    # donut example
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N//2) + [1]*(N//2))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = [] # keep track of log-likelihoods
    learning_rate = 0.00005
    regularization = 0.2
    last_error_rate = None
    for i in range(3000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = get_log_likelihood(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 300 == 0:
            print("i:", i, "ll:", ll, "classification rate:", 1 - er)
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    test_xor()
    # test_donut()