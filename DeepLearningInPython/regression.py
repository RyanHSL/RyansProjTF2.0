import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#generate and plot the data
N = 500
X = np.random.random((N, 2))*4 - 2 #in between (-2, 2)
Y = X[:, 0]*X[:, 1] #makes a saddle shape

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

#make a neural network and train it
D = 2
M = 100 #number of hidden units

#define the weight and the bias of the first layer
W = np.random.randn(D, M)/np.sqrt(D)
b = np.zeros(M)
#define the weight and the bias of the second layer
V = np.random.randn(M)/np.sqrt(M)
c = 0
#define the forward function which returns Z and Yhat using linear function and relu
def forward(X):
    Z = X.dot(W) + b
    Z = Z*(Z>0)#relu
    yhat = Z.dot(V) + c
    return Z, yhat
#define the derivative function of V
def derivative_v(Z, Y, Yhat):
    return (Y - Yhat).dot(Z)
#define the derivative function of c
def derivative_c(Y, Yhat):
    return (Y - Yhat).sum(axis = 0)
#define the derivative function of W
def derivative_w(X, Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V)*(Z>0)
    return X.T.dot(dZ)
#define the derivative function of b
def derivative_b(Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V)*(Z>0)
    return dZ.sum(axis = 0)
#define the update function which does the gradient descent of V, c, W, b and returns W, b, V, c
def update(X, Z, Y, Yhat, W, b, V, c, learning_rate = 1e-4):
    V += learning_rate*derivative_v(Z, Y, Yhat)
    c += learning_rate*derivative_c(Y, Yhat)
    W += learning_rate*derivative_w(X, Z, Y, Yhat, V)
    b += learning_rate*derivative_b(Z, Y, Yhat, V)
    return W, b, V, c
#define the get_cost function which returns the mean squared error
def get_cost(Y, Yhat):
    return ((Y - Yhat)**2).mean()
#run the training loop, plot the costs and plot the final result
#define the cost array
costs = []
#run through 200 epochs
for i in range(200):
    #use forward function to get the Z and Yhat
    Z, Yhat = forward(X)
    #use the update function to get the W, b, V, c
    W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)
    #use get_cost function to get the cost
    cost = get_cost(Y, Yhat)
    #append the cost to the cost array
    costs.append(cost)
    #for every 25 elements print the cost
    if i%25 == 0:
        print("Cost: ", cost)
#plot the cost array
plt.plot(costs)
plt.show()
#plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y)
#surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, Yhat = forward(Xgrid)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth = 0.2, antialiased = True)
plt.show()

#plot magnitude of residuals
Ygrid = Xgrid[:,0]*Xgrid[:,1]
R = np.abs(Ygrid - Yhat)

plt.scatter(Xgrid[:,0], Xgrid[:,1], c = R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth = 0.2, antialiased = True)
plt.show()
