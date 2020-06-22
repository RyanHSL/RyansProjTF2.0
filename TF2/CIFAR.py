from tensorflow.keras.layers import Input, Conv2D, Flatten, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Load in the cifar10 data, preprocess the training and test data. The target data in cifar10 is (K,1) so I need to flatten it
#Print the X shape and Y shape
cifar10 = tf.keras.datasets.cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
Y_train, Y_test =
#get the number of output classes

#Build the model using functional API
#If input has 1 colour channel, the filter size is (1, kernal_width, kernal_height, feature_map)
#If input has 3 colour channels, the filter size is (3, kernal_width, kernal_height, feature_map)

#Compile and fit. Loss function is sparse_categorical_crossentropy

#Plot loss per iteration

#Plot accuracy per iteration

#Plot confusion matrix

#Label mapping

#Show some msiclassified examples
#TODO: add label names
