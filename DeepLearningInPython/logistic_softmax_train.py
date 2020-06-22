import numpy as np
import matplotlib.pyplot as plt

from process import get_data
from sklearn.utils import shuffle

#y2Indicator function to create an indicator matrix
def y2Indicatior(y, K):
    N = len(y)
    indicator = np.zeros((N, K))
    for i in range(N):
        indicator[i, y[i]] = 1
    return indicator
#get X, Y from data, shuffle them, change Y type to int32, get number of input features and outputs, which are D and K
X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))
#seperate the data: training data are the last 100 samples, test data are the rest.
X_train = X[:-100]
Y_train = Y[:-100]
X_test = X[-100:]
Y_test = Y[-100:]
#Initialize the Y train indicator using y2Indicator function and initialize the Y test indicator using y2Indicator
Y_train_indicator = y2Indicatior(Y_train, K)
Y_test_indicator = y2Indicatior(Y_test, K)
#Randomize the weights and create b as a single zero column
W = np.random.randn(D, K)
b = np.zeros(K)
#define softmax function
def softmax(a):
    return np.exp(a)/np.exp(a).sum(axis = 1, keepdims = True)
#define a forward function
def forward(X, W, b):
    return softmax(X.dot(W) + b)
#define predict function: pick the max argument in the parameter list
def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis = 1)
#define the classification rate function
def classification_rate(Y, P):
    # total_num = 0
    # correct_num = 0
    # for i in range(len(Y)):
    #     if Y[i] == T[i]:
    #         correct_num += 1
    #     total_num += 1
    # return correct_num/total_num
    return np.mean(Y==P)
#define the cross entropy function to calculate the cost
def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY)) #mean = sum*constant, minimizing the sum of cross_entropy times a positive constanst is the same as minimizing the sum of cross_entropy
#Initialize the train cost array and test cost array and learning rate
train_costs = []
test_costs = []
learning_rate = 0.001
#run through 10000 epochs, append the train cost and test cost to the train cost and test cost arrays, do gradient descents of W and b and print train cost and test cost every 1000 iterations
for i in range(10000):
    pYtrain = forward(X_train, W, b)
    pYtest = forward(X_test, W, b)

    ctrain = cross_entropy(Y_train_indicator, pYtrain)
    ctest = cross_entropy(Y_test_indicator, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W -= learning_rate*X_train.T.dot(pYtrain - Y_train_indicator)
    b -= learning_rate*(pYtrain - Y_train_indicator).sum(axis = 0)
    if i%1000 == 0:
        print("The train cost is ", ctrain, " and the test cost is ", ctest)
#print the final train and final test classification rate
print("The final train classification rate is ", classification_rate(Y_train, predict(pYtrain)))
print("The final test classification rate is ", classification_rate(Y_test, predict(pYtest)))
#print the train cost array and test cost array
legend1 = plt.plot(train_costs, label = "train costs")
legend2 = plt.plot(test_costs, label = "test cost")
plt.legend([legend1, legend2])
plt.show()