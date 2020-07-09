from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import wget
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the movielens data
wget.download("http://files.grouplens.org/datasets/movielens/ml-20m.zip")
# Unzip the zip file
with zipfile.ZipFile("ml-20m.zip", "r") as zip_ref:
    zip_ref.extractall()
"""
Uncomment the code above if you do not have those files
"""
# Read 'ratings.csv' from the ml-20m directory and print the head of the data
df = pd.read_csv("ml-20m/ratings.csv")
# print(df.head())
# I cannot trust the userId and movieId to be numbered 1...N-1
# So I will set my own ids
#
# current_user_id = 0
# custom_user_map = {}
# def map_user_id(row):
#     global current_user_id, custom_user_map
#     old_user_id = row['userId']
#     if old_user_id not in custom_user_map:
#         custom_user_map[old_user_id] = current_user_id
#         current_user_id += 1
#     return custom_user_map[old_user_id]
# df['new_user_id'] = df.apply(map_user_id, axis=1)
df.userId = pd.Categorical(df.userId)
df["new_user_id"] = df.userId.cat.codes
# print(df.head())
# Do the same thing for movie ids
# current_movie_id = 0
# custom_movie_map = {}
# def map_movie_id(row):
#     global current_movie_id, custom_movie_map
#     old_movie_id = row["movieId"]
#     if old_movie_id not in custom_movie_map:
#         custom_movie_map[old_movie_id] = current_movie_id
#         current_movie_id += 1
#     return custom_movie_map[old_movie_id]
# df["movieId"] = df.apply(map_movie_id, axis=1)
df.movieId = pd.Categorical(df.movieId)
df["new_movie_id"] = df.movieId.cat.codes
print(df.head())
# Get user_ids , movie_ids, and ratings as separate arrays
user_ids = df["new_user_id"].values
movie_ids = df["new_movie_id"].values
ratings = df["rating"].values
# Get number of users and number of movies
uNum = len(set(user_ids))
mNum = len(set(movie_ids))
# Set embedding dimension
K = 20
# Make a neural network
# Create a user input layer
u = Input(shape=(1, ))
# Create a movie input layer
m = Input(shape=(1, ))
# User embedding (the output is (num_samples, 1, K))
uEmb = Embedding(uNum, K)(u)
# Movie embedding (the output is (num_samples, 1, K))
mEmb = Embedding(mNum, K)(m)
# Flatten both embeddings (the output is (num_samples, K))
uFlat = Flatten()(uEmb)
mFlat = Flatten()(mEmb)
# Concatenate user-movie embeddings into a feature vector (the output is (num_samples, 2K))
x = Concatenate()([uFlat, mFlat])
# Now I have a feature vector. Construct a regular ANN
x = Dense(512, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
x = Dense(1)(x)
# Build the model and compile (the model has two inputs)
model = Model(inputs=[u, m], outputs=x)
model.compile(optimizer=SGD(lr=1e-2, momentum=3e-1),
              loss="mse")
# Shuffle and split the data
user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)
brk = int(0.8*len(user_ids))
user_train = user_ids[:brk]
movie_train = movie_ids[:brk]
ratings_train = ratings[:brk]

user_test = user_ids[brk:]
movie_test = movie_ids[brk:]
ratings_test = ratings[brk:]
# Center the ratings (Normalization without dividing standard deviation)
mRatings = np.mean(ratings)
ratings_train -= mRatings
ratings_test -= mRatings
# Fit the model the data is an array of users and movies
r = model.fit([user_train, movie_train], ratings_train, batch_size=1024, epochs=25, validation_data=([user_test, movie_test], ratings_test))
# Plot the loss
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend()
plt.show()
# Calculate the final loss (Note: take the square root since the loss is mean squared error)
print(np.sqrt(0.6840))

model.save("Recommender_System.h5")