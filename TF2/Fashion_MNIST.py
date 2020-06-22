from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools

#Load the fashion_mnist data, preprocess them then print out the X_train shape
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
print(X_train.shape)
#Reduce the dimension by 1 at the end(using -1 to indicate the position) in order to make the data 3D, then print out the X_train shape
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print(X_train.shape)
#Get the number of output classes and print it
K = len(set(Y_test))
print("Number of Output Classes: ", K)
#Build the model using functional API
i = Input(shape = X_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation="relu")(i)
x = Conv2D(64, (3, 3), strides=2, activation="relu")(x)
x = Conv2D(128, (3, 3), strides=2, activation="relu")(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
x = Dense(K, activation="softmax")(x)

model = Model(i, x) #The first parameter means input and the last paramter means output
#Compile and fit. Loss is sparse_categorical_crossentropy
model.compile(optimizer = "adam",
          loss = "sparse_categorical_crossentropy",
          metrics = ["accuracy"])
r = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
#Plot the loss per iteration
plt.plot(r.history["loss"], label = "loss")
plt.plot(r.history["val_loss"], label = "val loss")
plt.legend()
plt.show()
#Plot accuracy per iteration
plt.plot(r.history["accuracy"], label = "accuracy")
plt.plot(r.history["val_accuracy"], label = "val accuracy")
plt.legend()
plt.show()
#Plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize = False, title = "Confusion matrix", cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normailization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

p_test = model.predict(X_test).argmax(axis = 1)
cm = confusion_matrix(Y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))
#Label mapping
labels = '''T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split("\n")
#Show the missclassed samples
misclassified_idx = np.where(p_test != Y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(X_test[i].reshape(28, 28), cmap = "gray")
plt.title("True label: %s Predicted: %s" % (labels[Y_test[i]], labels[p_test[i]]))