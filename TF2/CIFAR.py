from tensorflow.keras.layers import Input, Conv2D, Flatten, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools

#Load in the cifar10 data, preprocess the training and test data. The target data in cifar10 is (K,1) so I need to flatten it
#Print the X shape and Y shape
cifar10 = tf.keras.datasets.cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
Y_train, Y_test = Y_train.flatten(), Y_test.flatten()
print("X shape: ", X_train.shape)
print("Y shape: ", Y_train.shape)
#get the number of output classes
K = len(set(Y_train))
print("The number of output classes is: ", K)
#Build the model using functional API
#If input has 1 colour channel, the filter size is (1, kernal_width, kernal_height, feature_map)
#If input has 3 colour channels, the filter size is (3, kernal_width, kernal_height, feature_map)
i = Input(shape=X_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(i)
x = Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
#x = Dense(1024, activation="relu")(x)
x = Dense(K, activation="softmax")(x)
#This model overfits
model = Model(i, x)
#Compile and fit. Loss function is sparse_categorical_crossentropy
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=15)
#Plot loss per iteration
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend
plt.show()
#Plot accuracy per iteration
plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="accuracy")
plt.legend
plt.show()
#Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
     """
     This function prints and plots the confusion matrix.
     Normalization can be applied by setting `normalize=True`.
     """
     if normalize:
         cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
         print("Normalized confusion matrix")
     else:
         print("Confusion matrix, without normalization")

     print(cm)

     plt.imshow(cm, interpolation="nearest", cmap = cmap)
     plt.title(title)
     plt.colorbar()
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation = 45)
     plt.yticks(tick_marks, classes)

     fmt = ".2f" if normalize else "d"
     thresh = cm.max()/2
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment = "center",
                  color = "white" if cm[i, j] > thresh else "black")

     plt.tight_layout()
     plt.ylabel("True label")
     plt.xlabel("Predicted label")
     plt.show()

p_test = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(Y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))
#Label mapping
labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()
#Show some msiclassified examples
# TODO: add label names
misclassified_idx = np.where(p_test != Y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(X_test[i], cmap = "gray")
plt.title("True label: %s Predicted: %s" % (labels[Y_test[i]], labels[p_test[i]]));
