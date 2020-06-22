import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, W2, b1, b2):
    Z = 1/(1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA/expA.sum(axis = 1, keepdims = True)
    return Y, Z

def cost(T, Y):
    #log likelihood of (softmax)categorical cross entropy
    cost = T*np.log(Y)
    #return the sum of log likelihood
    return cost.sum()

def derivative_w2(Z, T, Y):
    #get the shape from targets
    # N, K = T.shape
    #get the number of hidden units
    # M = Z.shape[1]
    #slow way to get the derivation
    # ret1 = np.zeros((M, K))
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             ret1[m, k] += (T[n, k] - Y[n, k])*Z[n, m]
    ret1 = Z.T.dot(T-Y)
    return ret1

def derivative_b2(T, Y):
    return (T - Y).sum(axis = 0)

def derivative_w1(X, Z, T, Y, W2):
    # N, D = X.shape
    # M, K = W2.shape

    #slow
    # ret1 = np.zeros((D, M))
    # for n in range(N):
    #     for d in range(D):
    #         for k in range(K):
    #             for m in range(M):
    #                 ret1[d, m] += (T[n, k] - Y[n, k])*W2[m, k]*Z[n, m]*(1 - Z[n, m])*X[n, d]
    ret1 = (T - Y).dot(W2.T)*Z*(1 - Z)
    return X.T.dot(ret1)

def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2)*Z*(1 - Z)).sum(axis = 0)

def classification_rate(Y, P):
    total_num = 0
    correct_num = 0
    for i in range(len(Y)):
        if Y[i] == P[i]:
            correct_num += 1
        total_num += 1
    return float(correct_num)/total_num

def main():
    #create the data: Nclass, dimension of input, hidden layer size, number of classes
    Nclass = 500
    D = 2
    M = 3
    K = 3
    #randomly initialize classes and stack them together
    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])
    #initialize Y
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)
    #turn the target to an indicator variable
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1
    #plot the data
    plt.scatter(X[:, 0], X[:, 1], c = Y, s = 100, alpha= 0.5)
    plt.show()
    #randomly initialize the weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)
    #set learning rate
    learning_rate = 10e-7
    #set the cost function as an array
    costs = []
    #loop through 100000 epochs and use the forward function to get the outputs and hidden layers. For each 100 epochs, get the cost and prediction and use classification_rate function
    #to calculate the accuracy. Print the cost and accuracy. Append the cost to the cost array
    for epoch in range(1000000):
        output, hidden = forward(X, W1, W2, b1, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis = 1)
            acc = classification_rate(Y, P)
            print("The cost is ", c, "and the accuracy is ", acc)
            costs.append(c)
    #do the gradient ascent using function derivative_w2, derivative_b2, derivative_w1, derivative_b1
        W2 += learning_rate*derivative_w2(hidden, T, output)
        b2 += learning_rate*derivative_b2(T, output)
        W1 += learning_rate*derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate*derivative_b1(T, output, W2, hidden)
    #plot the cost array
    plt.plot(costs)
    plt.show()

if __name__ == "__main__":
        main()