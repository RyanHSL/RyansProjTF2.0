import numpy as np
import pandas as pd

#define a get_data function to read the data, turn data into a numpy matrix, split out X and Y(Y is the last column)
#and normalize all numerical columns then work on the categorical column(time of day) doing one-hot encoding
def get_data():
    fl = pd.read_csv("ecommerce_data.csv", encoding='cp1252')
    data = fl.values
    # data = np.genfromtxt("ecommerce_data.csv", delimiter=",")
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1]

    X[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean())/X[:, 2].std()

    N, D = X.shape
    X2 = np.zeros((N, D+3)) # +3 = -1 + 4
    X2[:, 0:D-1] = X[:, 0:D-1]

    for n in range(N):
        t = int(X[n, D-1])
        X2[n, D-1+t] = 1

    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D - 1].astype(np.int32)] = 1
    assert(abs(Z - X2[:, -4:]).sum() <= 1e-10)
    return X2, Y
#define a get_binary_data function to call get_data function and filter it by only taking 1 or 0
def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2
