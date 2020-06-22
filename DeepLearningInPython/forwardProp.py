import numpy as np
import matplotlib.pyplot as plt

#define Number of class samples: 500
Nclass = 500
#create 3 gaussian clouds
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])
#create y as a numpy array
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
#visualize the data
plt.scatter(X[:, 0], X[:, 1], c = Y, s = 100, alpha = 0.5)
plt.show()
#define input size, hidden layer size and output size
D = 2
M = 3
K = 3
#initialize the weights and biases
W1 = np.random.randn(D, M)
B1 = np.random.randn(M)
W2 = np.random.randn(M, K)
B2 = np.random.rand(K)
#define the forward action. parameters: input, weights and biases
def forward(X, W1, W2, B1, B2):
    Z = 1/(1 + np.exp(-X.dot(W1) - B1))
    A = Z.dot(W2) + B2
    expA = np.exp(A)
    Y = expA/expA.sum(axis = 1, keepdims = True)
    return Y
#calculate the classification rate. parameters: target, prediction
def classiffication_rate(Y, P):
    total_num = 0
    correct_num = 0
    for i in range(len(Y)):
        total_num += 1
        if Y[i] == P[i]:
            correct_num += 1
    accuracy = float(correct_num)/total_num
    return accuracy
#call the forward function and calculate the prediction and classification rate
P_Y_given_X = forward(X, W1, W2, B1, B2)
P = np.argmax(P_Y_given_X, axis = 1)

assert(len(P) == len(Y))

print("The classification rate of randomly chosen weights is:", classiffication_rate(Y, P))
