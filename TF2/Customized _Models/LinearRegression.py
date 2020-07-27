import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(tf.keras.Model):
    # Create two variables W and b.
    # Use random normal initializer to create W. The shape is (num_inputs, num_outputs)
    # b will be a vector of zeros with length num_outputs
    # Save each of these variables to a list called params
    def __init__(self, num_inputs, num_outputs):
        super(LinearRegression, self).__init__()
        self.W = tf.Variable(tf.random_normal_initializer()((num_inputs, num_outputs)))
        self.b = tf.Variable(tf.zeros(num_outputs))
        self.params = [self.W, self.b]

        return
    # Create a call function to do matrix multiplication between the inputs and the W then plus b
    def call(self, inputs):
        prediction = tf.matmul(inputs, self.W) + self.b

        return prediction

# Create the dataset
N, D, K = 100, 1, 1
X = np.random.random((N, D)) * 2 - 1 # Make it randomly distributes between -1 and 1
w = np.random.randn(D, K) # True parameter which model does not know
b = np.random.randn() # True parameter which model does not know
Y = X.dot(w) + b + np.random.randn(N, 1) * 0.1
# Important: Because tensorflow creates model parameters as floats by default while numpy creates
# the data as doubles by default, I need to cast the numpy data to float32. It was float64.
X = X.astype(np.float32)
Y = Y.astype(np.float32)
# Define the loss MSE
def get_loss(model, inputs, targets):
    prediction = model(inputs)
    loss = tf.reduce_mean(tf.square(targets - prediction))

    return loss
# Gradient Function
def get_grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_val = get_loss(model, inputs, targets)
    grad = tape.gradient(loss_val, model.params)

    return grad
# Create and train the model
linear_model = LinearRegression(D, K)
# Print the parameters before training
print("Initial params:")
print(linear_model.W)
print(linear_model.b)
# Store the losses here
losses = []
# Create an optimizer SGD
optimizer = tf.keras.optimizers.SGD(lr=0.2)
# Run the training loop
for i in range(100):
    # Get gradients
    g = get_grad(linear_model, X, Y)
    # Do one step of gradient descent: param <- param - learning_rate * grad
    loss = get_loss(linear_model, X, Y)
    optimizer.apply_gradients(zip(g, linear_model.params))
    # Store the loss
    losses.append(loss)
# Plot the losses
plt.plot(losses)
plt.show()
# Plot both the predicted values and the actual values
x_axis = np.linspace(X.min(), X.max(), 100)
y_axis = linear_model.predict(x_axis.reshape(-1, 1)).flatten()

plt.scatter(X, Y)
plt.plot(x_axis, y_axis)
plt.show()
# Print the trained parameters
print("Predicted parameters:")
print(linear_model.W)
print(linear_model.b)
# Print true parameters
print("True parameters:")
print(w, b)
