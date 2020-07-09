from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

import os
import requests
import shutil
import wget
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Uncomment the code below if you do not have the zip file
"""
# wget.download("https://lazyprogrammer.me/course_files/Food-5K.zip")
r = requests.get("https://lazyprogrammer.me/course_files/Food-5K.zip", stream=True)
if r.status_code == 200:
    with open("Food-5K.zip", 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)
with zipfile.ZipFile("Food-5K.zip", "r") as zipRef:
    zipRef.extractall()
plt.imshow(image.load_img("Food-5K/training/0_100.jpg"))
plt.show()

plt.imshow(image.load_img("Food-5K/training/1_100.jpg"))
plt.show()
"""
Uncomment the code below if you do not have those directories
"""
os.chdir("Food-5K")
os.mkdir("data")
os.mkdir("data/train")
os.mkdir("data/test")
os.mkdir("data/train/nonfood")
os.mkdir("data/train/food")
os.mkdir("data/test/nonfood")
os.mkdir("data/test/food")
"""
Uncomment the code below if you do not have not separated the training data and validation data
"""
# os.chdir("Food-5K")
path = "training"
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        if "0_" in file:
            shutil.move(f"{path}/{file}", "data/train/nonfood")
        elif "1_" in file:
            shutil.move(f"{path}/{file}", "data/train/food")

path = "validation"
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        if "0_" in file:
            shutil.move(f"{path}/{file}", "data/test/nonfood")
        elif "1_" in file:
            shutil.move(f"{path}/{file}", "data/test/food")
# Define the train_path and the valid_path
train_path = 'data/train'
valid_path = 'data/test'
# These images are pretty large and of different sizes
# Let me load them all in as the same (smaller) size [200, 200]
IMAGE_SIZE = [200, 200]
# get all training images and validation images
train_images = glob(train_path + "/*/*.jpg")
valid_images = glob(valid_path + "/*/*.jpg")
# # get number of classes
folders = glob(train_path + "/*")
K = len(folders)
print(folders)
# Randomly display an image using from train_images np.random.choice
plt.imshow(image.load_img(np.random.choice(train_images)))
plt.show()
# Build the pretrained model. input_shape should be the dimension of the image plus colour channels
# Since this is a pretrained model, the weight should be "imagenet"
# Include_top should be false since I will build the top ANN model including Flatten and Dense layers
ptm = PretrainedModel(input_shape=IMAGE_SIZE + [3],
                      weights="imagenet",
                      include_top=False)
# Freeze pretrained model weights via setting model trainable to false
ptm.trainable = False
# Map the data into features vectors. The pretrained model has input and output objects
# Keras image data generator returns classes one-hot coded so I use softmax at the output layer and I only need a K-output layer
x = Flatten()(ptm.output)
x = Dense(K, activation="softmax")(x)
# # Create a model object
model = Model(ptm.input, x)
# view the structure of the model
print(model.summary())
# Create an instance of ImageDataGenerator.(Note: the preprocessing_function shoud be preprocess_input)
gen_train = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               preprocessing_function=preprocess_input)
gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
# Declare the batch_size
batch_size = 128
# Create the train_generator and valid_generator using flow_from_directory
train_generator = gen_train.flow_from_directory(train_path,
                                                target_size=IMAGE_SIZE,
                                                shuffle=True,
                                                batch_size=batch_size)
val_generator = gen_val.flow_from_directory(valid_path,
                                            target_size=IMAGE_SIZE,
                                            batch_size=batch_size)
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Fit the model using fit_generator. step_per_epoch is number of image files divided by batch_size
print(int(np.ceil(len(valid_images) / batch_size)))
r = model.fit_generator(train_generator, validation_data=val_generator, epochs=10,
                        steps_per_epoch=int(np.ceil(len(train_images) / batch_size)),
                        validation_steps=int(np.ceil(len(valid_images) / batch_size)))
# Create a 2nd train generator without data augmentation to get the true train accuracy
train_generator2 = gen_train.flow_from_directory(train_path,
                                                 target_size=IMAGE_SIZE,
                                                 batch_size=batch_size)
# Evaluate the generator
model.evaluate_generator(train_generator, steps=int(np.ceil(len(train_images)/batch_size)))
# Plot the loss
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend()
plt.show()
# Plot the accuracy
plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="val accuracy")
plt.legend()
plt.show()

os.chdir("..")
model.save("food_classifier.h5")