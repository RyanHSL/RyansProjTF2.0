from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools

#Load in the keras mnist dataset. Split them into train sets and test sets.
#Preprocess the features and print the shape of training data
(X_train, Y_train),(X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
print("Shape of X_train: ", X_train.shape)
#Build the model. Flatten layer, Dense layer, dropout layer and outout dense layer.
models = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])
#Compile the model using sparse_categorical_corssentropy because the output is one hot encoding
models.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])
#Train the model
r = models.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)
#Plot loss per iteration
plt.plot(r.history["loss"], label = "Train Loss")
plt.plot(r.history["val_loss"], label = "Test Loss")
plt.legend()
plt.show()
#Plot accuracy per iteration
plt.plot(r.history["accuracy"], label = "Train Accuracy")
plt.plot(r.history["val_accuracy"], label = "Test Accuracy")
plt.legend()
plt.show()
#Evaluate the model
print("Evaluate the model: ", models.evaluate(X_test, Y_test))
#Plot confusion function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
     """
     This function prints and plots the confusion matrix.
     Normalization can be applied by setting `normalize=True`.
     """
     if normalize:
         cm = cm.astype("float")/cm.sum(axis=1)[:, np.newaxis]
         print("Normalized confusion matrix")
     else:
         print("Confusion matrix, without normalization")

     print(cm)

     plt.imshow(cm, interpolation="nearest", cmap = cmap)
     plt.title(title)
     plt.colorbar()
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation=45)
     plt.yticks(tick_marks, classes)

     fmt = '.2f' if normalize else 'd'
     thresh = cm.max()/2
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment = "center",
                    color="white" if cm[i, j]>thresh else "black")

         plt.tight_layout()
         plt.ylabel("True label")
         plt.xlabel("Predicted label")
         plt.show()

p_test = models.predict(X_test).argmax(axis = 1)
cm = confusion_matrix(Y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))