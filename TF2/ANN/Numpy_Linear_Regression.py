from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


class Model:
    def __init__(self):
        self.layers = []
        self.loss = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        # Forward pass
        for i, _ in enumerate(self.layers):
            forward = self.layers[i].forward(X)
            X = forward

        return forward

    def train(
            self,
            X_train,
            Y_train,
            learning_rate,
            epochs,
            verbose=False
    ):
        for epoch in range(epochs):
            loss = self._run_epoch(X_train, Y_train, learning_rate)

            if verbose:
                if epoch % 50 == 0:
                    print(f'Epoch: {epoch}. Loss: {loss}')

    def _run_epoch(self, X, Y, learning_rate):
        # Forward pass
        for i, _ in enumerate(self.layers):
            forward = self.layers[i].forward(input_val=X)
            X = forward

        # Compute loss and first gradient
        bce = BinaryCrossEntropy(forward, Y)
        error = bce.forward()
        gradient = bce.backward()

        self.loss.append(error)

        # Backpropagation
        for i, _ in reversed(list(enumerate(self.layers))):
            if self.layers[i].type != 'Linear':
                gradient = self.layers[i].backward(gradient)
            else:
                gradient, dW, dB = self.layers[i].backward(gradient)
                self.layers[i].optimize(dW, dB, learning_rate)

        return error


class Layer:
    """Layer abstract class"""

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def optimize(self):
        pass


class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim, 1)
        self.type = 'Linear'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = input_val
        return np.matmul(self.weights, input_val) + self.biases

    def backward(self, dA):
        dW = np.dot(dA, self._prev_acti.T)
        dB = dA.mean(axis=1, keepdims=True)

        delta = np.dot(self.weights.T, dA)

        return delta, dW, dB

    def optimize(self, dW, dB, rate):
        self.weights = self.weights - rate * dW
        self.biases = self.biases - rate * dB


class ReLU(Layer):
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'ReLU'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = np.maximum(0, input_val)
        return self._prev_acti

    def backward(self, dJ):
        return dJ * np.heaviside(self._prev_acti, 0)


class Sigmoid(Layer):
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'Sigmoid'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = 1 / (1 + np.exp(-input_val))
        return self._prev_acti

    def backward(self, dJ):
        sig = self._prev_acti
        return dJ * sig * (1 - sig)


class MeanSquaredError(Layer):
    def __init__(self, predicted, real):
        self.predicted = predicted
        self.real = real
        self.type = 'Mean Squared Error'

    def forward(self):
        return np.power(self.predicted - self.real, 2).mean()

    def backward(self):
        return 2 * (self.predicted - self.real).mean()


class BinaryCrossEntropy(Layer):
    def __init__(self, predicted, real):
        self.real = real
        self.predicted = predicted
        self.type = 'Binary Cross-Entropy'

    def forward(self):
        n = len(self.real)
        loss = np.nansum(-self.real * np.log(self.predicted) - (1 - self.real) * np.log(1 - self.predicted)) / n

        return np.squeeze(loss)

    def backward(self):
        n = len(self.real)
        return (-(self.real / self.predicted) + ((1 - self.real) / (1 - self.predicted))) / n

def numpy_expression():
    w, b = 0, 0
    epochs = 200
    learning_rate = 1e-3

    for e in range(epochs):
        yhat = xs*w + b
        loss = np.mean(np.square(ys - yhat))
        dw, db = (ys - yhat).dot(xs), (ys - yhat).sum()
        w -= learning_rate*dw
        b -= learning_rate*db

        if e % 20 == 0:
            plt.cla()
            plt.scatter(xs, ys)
            plt.scatter(xs, yhat)
            plt.show()
            # plt.next(0.5, 0, "Loss=%.4f" %loss, fontdict={"size":20, "color":"red"})
            # plt.pause(0.1)

    print(w, b)

    return


def generate_data(samples, shape_type='circles', noise=0.05):
    # We import in the method for the shake of simplicity
    import matplotlib
    import pandas as pd

    from matplotlib import pyplot as plt
    from sklearn.datasets import make_moons, make_circles
    if shape_type is 'moons':
        X, Y = make_moons(n_samples=samples, noise=noise)
    elif shape_type is 'circles':
        X, Y = make_circles(n_samples=samples, noise=noise)
    else:
        raise ValueError(f"The introduced shape {shape_type} is not valid. Please use 'moons' or 'circles' ")

    data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))

    return data


def plot_generated_data(data):
    ax = data.plot.scatter(x='x', y='y', figsize=(16, 12), color=data['label'],
                           cmap=matplotlib.colors.ListedColormap(['skyblue', 'salmon']), grid=True);

    return ax

def solution_model():
    data = generate_data(samples=5000, shape_type='circles', noise=0.04)
    plot_generated_data(data);
    X = data[['x', 'y']].values
    Y = data['label'].T.values

    # Create model
    model = Model()

    # Add layers
    model.add(Linear(2, 5))
    model.add(ReLU(5))

    model.add(Linear(5, 2))
    model.add(ReLU(2))

    model.add(Linear(2, 1))
    model.add(Sigmoid(1))

    # Train model
    model.train(X_train=X.T,
                Y_train=Y,
                learning_rate=0.05,
                epochs=9000,
                verbose=True)
    plt.figure(figsize=(17, 10))
    plt.plot(model.loss)

    from sklearn.metrics import roc_auc_score

    # Make predictions
    predictions = model.predict(X.T).T

    # Format the predictions
    new_pred = []

    for p in predictions:
        if p < 0.5:
            new_pred.append(0)
        else:
            new_pred.append(1)

    # Calculate the score
    roc_auc_score(y_true=Y, y_score=new_pred)

    return model

if __name__ == "__main__":
    model = solution_model()

    # model.save("Models/numpy_linear.h5")