import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the data
data = pd.read_csv("Moore.csv").values
X = data[:,0].reshape(-1, 1) #Make it a 2-D array of size N*D where D = 1
Y = data[:,1]
#plot the data
plt.scatter(X, Y)
plt.show()
#since we want a linear model, take the log and scatter plot the data
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()
#Center the X so that the mean is 0
#We could scale it too, but we need to reverse the transform later
X = X - X.mean()
#Create the Tensorflow model. Since it is linear regression, we don't need to write the activation function
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (1,)),
    tf.keras.layers.Dense(1)
])
#Compile the model using SGD optimizer object with learning rate 0.001 and momentum 0.9. Loss function is mse
model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss="mse")
#Define the learning rate scheduler
def schedule(epoch, lr):
    #multiply the learning rate by 10 if epoch is greater than 50
    if epoch >= 50:
        return 0.0001
    return 0.001
#build the scheduler using tensorflow.keras.callbacks.LearningRateScheduler(schedule)
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
#Train the model using 200 epochs and scheduler
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])
#Plot the loss
plt.figure(1)
plt.plot(r.history["loss"], label = "loss")
#Get the slope of the line
#The slope of the line is related to doubling rate of transistor count
print(model.layers)
print(model.layers[0].get_weights())

a = model.layers[0].get_weights()[0][0,0]
#Print time to double
print("Time to double: ", np.log(2)/a)
#The analytical solution
X = np.array(X).flatten()
Y = np.array(Y)
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator
print(a, b)
print("Time to double: ", np.log(2) / a)

#Make predictions and print the predictions
p = model.predict(X)
print(p)
#Round to get the actual predictions and print them
#Note: has to be flattened since the targets are size(N,) while the predictions are size (N,1)
intp = np.round(p).flatten()
print(intp)
#Calculate the accuracy, compare it to model.evaluate() output by printing them
print("Model Accuray: ", np.mean(p == Y))
print("Evaluate the Model: ", model.evaluate(X, Y))
#Make sure the line fits our data by scatter plotting the Y values and plotting the Yhat values
Yhat = model.predict(X).flatten()
plt.figure(2)
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
#Mannual calculation
#Get the weights of the first layer
w, b = model.layers[0].get_weights()
#Reshape X because we flattened it again earlier
X = np.reshape(-1, 1)
#Calculate the Yhat2 and flatten it
Yhat2 = (X.dot(w) + b).flatten()
#Use numpy.allclose() to compare Yhat and Yhat2.
#Don't use == for floating points
print("Comparison Result: ", np.allclose(Yhat, Yhat2))

# model.save("Linear_regression.h5")
# model = tf.keras.models.load_model("Linear_regression.h5")
# print(model.layers)
#model.evaluate(X, Y)