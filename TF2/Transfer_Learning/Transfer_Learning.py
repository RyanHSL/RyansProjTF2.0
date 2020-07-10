from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import os
import requests
import shutil
import wget
import zipfile
import numpy as np
import matplotlib.pyplot as plt

# Download the data and move them to different directories
# r = requests.get("https://lazyprogrammer.me/course_files/Food-5K.zip", stream=True)
# if r.status_code == 200:
#     with open("Food-5K.zip", "wb") as f:
#         r.raw.decode_content = True
#         shutil.copyfileobj(r.raw, f)
# with zipfile.ZipFile("Food-5K.zip", "r") as zipRef:
#     zipRef.extractall()
#
os.chdir("Food-5K")
# os.mkdir("data")
# os.mkdir("data/train")
# os.mkdir("data/test")
# os.mkdir("data/train/nonfood")
# os.mkdir("data/train/food")
# os.mkdir("data/test/nonfood")
# os.mkdir("data/test/food")
#
# path = "training"
# for file in os.listdir(path):
#     if os.path.isfile(os.path.join(path, file)):
#         if "0_" in file:
#             shutil.move(f"{path}/{file}", "data/train/nonfood")
#         elif "1_" in file:
#             shutil.move(f"{path}/{file}", "data/train/food")
#
# path = "validation"
# for file in os.listdir(path):
#     if os.path.isfile(os.path.join(path, file)):
#         if "0_" in file:
#             shutil.move(f"{path}/{file}", "data/test/nonfood")
#         elif "1_" in file:
#             shutil.move(f"{path}/{file}", "data/test/food")
# Define the train_path and the valid_path
train_path = "data/train"
valid_path = "data/test"
# These images are pretty big and of different sizes
# Let's load them all in as the same (smaller) size
IMAGE_SIZE = [200, 200]
# Create train_data and valid_data arrays
train_data = glob(train_path + "/*/*.jpg")
valid_data = glob(valid_path + "/*/*.jpg")
folders = glob("data/*")
K = len(folders)
# Build the pretrained model
ptm = PretrainedModel(input_shape=(IMAGE_SIZE + [3]),
                      weights="imagenet",
                      include_top=False)
# Map the data into feature vectors (Note: only need a Flatten layer)
x = Flatten()(ptm.output)
# Create the model object
model = Model(inputs=ptm.input, outputs=x)
# Print the model summary
print(model.summary())
# Create an instance of ImageDataGenerator which only needs the preprocessing_function parameter
gen_train = ImageDataGenerator(preprocessing_function=preprocess_input)
gen_valid = ImageDataGenerator(preprocessing_function=preprocess_input)
# Declare the batch size
batch_size = 128
# Create the generators
# (Nots: do not need to shuffle the train data because all I am is transforming it. The class_mode is "binary")
train_generator = gen_train.flow_from_directory(train_path,
                                                IMAGE_SIZE,
                                                batch_size=batch_size,
                                                class_mode="binary")
valid_generator = gen_valid.flow_from_directory(valid_path,
                                                IMAGE_SIZE,
                                                batch_size=batch_size,
                                                class_mode="binary")
# Declare the number of train_data and the number of the valid_data
Ntrain = len(train_data)
Nvalid = len(valid_data)
# Figure out the output size
feat = model.predict(np.random.random([1] + IMAGE_SIZE + [3]))
D = feat.shape[1]

X_train = np.zeros((Ntrain, D))
Y_train = np.zeros(Ntrain)
X_test = np.zeros((Nvalid, D))
Y_test = np.zeros(Nvalid)

# Populate the X_train and Y_train
i = 0
for x, y in train_generator:
    # get features using model.predict()
    feat = model.predict(x)
    # get the size of the batch which is the length of y
    # (may not always be batch_size)
    sz = len(y)
    # assign the features and y to X_train and Y_train
    X_train[i: i + sz] = feat
    Y_train[i: i + sz] = y
    # increment i
    i += sz
    print(i)
    # break the loop if i is greater or equal to Ntrain
    if i >= Ntrain:
        print("Break the loop")
        break
# Print i
print(i)
# Populate X_valid and Y_valid
i = 0
for x, y in valid_generator:
    feat = model.predict(x)
    sz = len(y)
    X_test[i: i + sz] = feat
    Y_test[i: i + sz] = y
    i += sz
    print(i)

    if i >= Nvalid:
        print("Break the loop")
        break

print(i)
# Normalize the X_train and X_valid since X_train.max() is much larger than X_train.min()
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Try the built-in logistic regression
# Create a LogisticRegression instance
logistic = LogisticRegression()
# Fit the normalized X_train and Y_train
logistic.fit(X_train, Y_train)
# Print the accuracies using LogisticRegression.score(X, Y)
print(logistic.score(X_train, Y_train))
print(logistic.score(X_test, Y_test))
# Do logistic regression in Tensorflow
# Build a logistic regression ANN model
i = Input(shape=(D, ))
x = Dense(1, activation="sigmoid")(i)
linearModel = Model(i, x)
# Compile and fit the model
linearModel.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
r = linearModel.fit(X_train, Y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, Y_test))
# Print the losses and accuracies
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