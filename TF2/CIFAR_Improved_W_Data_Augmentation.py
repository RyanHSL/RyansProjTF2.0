import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("CIFAR_Improved_WO_Data_Augmentation.h5")
print(model.layers)

cifar10 = tf.keras.datasets.cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
Y_train, Y_test = Y_train.flatten(), Y_test.flatten()
print("X shape is: ", X_train.shape)
print("Y shape is: ", Y_train.shape)

'''Adding this Data Augmentation will cause the validation loss/accuracy fluctuates 
That maybe caused by the learning rate or the batch size or the size of validation data'''

#Fit with data augmentation
#Note: If I run this after calling the previous model.fit(), it will continue training where it left off(Using the same weights and biases)
#Define the batch size
batch_size = 32
#Use ImageGenerator to create a data generator width_shift_range and height_shift_ranges are 0.1, horizaontal flips
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
#Use data generator flow(X_train, Y_train, batch_size) to get the train_generator
train_generator = data_generator.flow(X_train, Y_train, batch_size)
#Step per epoch is the shape of the first training data
steps_per_epoch = X_train.shape[0] // batch_size
#Fit the train generator
r = model.fit(train_generator, validation_data=(X_test, Y_test), steps_per_epoch=steps_per_epoch, epochs=50)

plt.plot(r.history["loss"], label="Loss")
plt.plot(r.history["val_loss"], label="Val Loss")
plt.legend
plt.show()

plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="Val accuracy")
plt.legend
plt.show()