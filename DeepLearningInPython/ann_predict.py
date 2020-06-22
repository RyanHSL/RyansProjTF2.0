import numpy as np
#import get_data from process.py
from process import get_data

#get data
X, Y = get_data()
#define sizes and randomly intialize weights. biases are zeros. 5 hidden units
M = 5
D = X.shape[1]
K = len(set(Y)) #the unique values of Y
W1 = np.random.randn(D, M)
B1 = np.zeros(M)
W2 = np.random.randn(M, K)
B2 = np.zeros(K)
#define softmax function
def softmax(a):
    expA = np.exp(a)
    return expA/expA.sum(axis = 1, keepdims = True)
#define a forward function
def forward(X, W1, B1, W2, B2):
    Z = np.tanh(X.dot(W1) + B1)
    return softmax(Z.dot(W2) + B2)
#calculate P(Y|X) and prediction
P_Y_given_X = forward(X, W1, B1, W2, B2)
predictions = np.argmax(P_Y_given_X, axis=1)
#define classification_rate function
def classification_rate(Y, P):
    return np.mean(Y == P)
#print prediction score
print("Accuracy: ", classification_rate(Y, predictions))