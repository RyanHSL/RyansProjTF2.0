import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D

#Make the dataset: N = 1000, X is randomly distributed between -3 and 3, Y = cos(2X[:,0]) + sin(2X[:,1])
N = 1000
X = np.random.random((N, 2))*6 - 3
Y = np.cos(2*X[:,0]) + np.sin(2*X[:,1]) #- np.cosh(2*X[:,0]) + np.sinh(2*X[:,1])
#Plot it using 3d projection, since I will print X[:,0], X[:,1] and Y
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], Y)
plt.show(ax)
#Build the model. Since it is linear regression it doesn't have activation function at output layer
models = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation="relu"),
    tf.keras.layers.Dense(1)
])
#Compile and fit. Notice: Since it is a linear regression model, I don't add accuracy metrics
optimizer = tf.keras.optimizers.Adam(0.01)
#loss = tf.keras.losses.MSE
models.compile(optimizer = optimizer, loss = "mse")
r = models.fit(X, Y, epochs=100)
#Plot the loss
plt.plot(r.history["loss"], label = "Loss")
plt.show()
#Plot the prediction surface using projection 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], Y)
plt.show(ax)
#Surface plot
line = np.linspace(-3, 3, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = models.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth = "0.2", antialiased = True)
plt.show()
#Can it extrapolate?
#Plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], Y)
plt.show(ax)
#Surface plot
line = np.linspace(-5, 5, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = models.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth = "0.2", antialiased = True)
plt.show()
