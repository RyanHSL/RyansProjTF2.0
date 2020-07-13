from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from skimage.io import imread

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load in the data and get the X_train, Y_train, X_test, Y_test
data = tf.keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = data.load_data()
# Map inputs to (-1, 1) for better training and print the shape of X_train
X_train, X_test = X_train / 255.0 * 2 - 1, X_test / 255.0 * 2 - 1
# Flatten the data and reshape the X_train and X_test
N, W, H = X_train.shape
D = W * H
X_train = X_train.reshape(-1, D)
X_test = X_test.reshape(-1, D)
# Declare the dimensionality of the latent space
latent_space = 100


# Build the generator model
# (Dense layers with LeakyRelu Activation. A BatchNormaliztion after each Dense Layer. D outputs)
# Use tanh activation at the output layer since the range is -1 to 1
def build_generator(latent_dim):
    i = Input(shape=(latent_dim,))
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(D, activation="tanh")(x)
    model = Model(i, x)

    return model


# Build the discriminator model using LeakyRelu activation. The img_size is the flatten size
def build_discriminator(img_size):
    i = Input(shape=(img_size,))
    x = Dense(512, activation="relu")(i)
    x = Dense(256, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(i, x)

    return model


"""Compile both models in preparation for training"""
# Build and compile the discriminator
# Create a build_discriminator instance with parameter flatten image size
discriminator = build_discriminator(D)
# Compile the discriminator instance
discriminator.compile(optimizer=Adam(2e-4, 0.5),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
# Create a build_generator instance with parameter latent_dim
generator = build_generator(latent_space)
# Create an input to represent noise sample from latent space
z = Input(shape=(latent_space,))
# Pass noise through generator to get an fake image
image = generator(z)
# Make sure only the generator is trained by freezing the weights
discriminator.trainable = False
# Pass the fake image to the discriminator then get the output.
# The true output is fake, but I label them real
fake_pred = discriminator(image)
# Create the combined model object.
# (Inputs: noise samples from latent space; Outputs: fake image output from discriminator instance)
combined_model = Model(z, fake_pred)
# Compile the combined model
combined_model.compile(optimizer=Adam(2e-4, 0.5),
                       loss="binary_crossentropy",
                       metrics=["accuracy"])
"""Train the GAN"""
# Set the Config
batch_size = 32
epochs = 30000
sample_period = 200  # every "sample_period" steps generate and save some data

# Create batch labels to use when calling train_on_batch
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)
# Create lists to store the losses of discriminator and generator
d_losses = []
g_losses = []
# Create a folder to store generated images
if not os.path.exists("gan_images"):
    os.mkdir("gan_images")

"""
A function to generate a grid of random samples from the generator
and save them to a file
"""


def sample_images(sample_epoch):
    # Declare the rows and columns which are 5 and 5
    rows, cols = 5, 5
    # Declare the noise object which is random normal with shape (rows*cols, latent_dim)
    sample_noise = np.random.randn(rows * cols, latent_space)
    # Get the image by using the generator to predict the noise
    img = generator.predict(noise)
    # Rescale images to 0 - 1 range. The original image range is -1 - 1
    img = img / 2.0 + 0.5
    # subplot the figure and axis
    fig, axs = plt.subplots(rows, cols)
    sample_idx = 0  # Counter. The final value should br 25(5*5)
    # Loop through all the rows and columns, reshape the gray scaled images at index idx then show it
    # Axis is off. Update the idx
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(img[sample_idx].reshape(H, W), cmap="gray")
            axs[i, j].axis("off")
            sample_idx += 1
    # Save the fig as ("gan_image/%d.png"%epoch)
    fig.savefig("gan_images/%d.png" % sample_epoch)
    # Close the plt
    plt.close()


"""
Main training loop
"""
for epoch in range(epochs):
    """Train the discriminator"""
    # Select a random batch of images.
    # Index is a random integer between 0 and X_train.shape[0], step size is the batch size
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_img = X_train[idx]
    # Generate fake images
    # Create a random normalized noise then use generator to predict the fake image
    noise = np.random.randn(batch_size, latent_space)
    fake_img = generator.predict(noise)
    # Train the discriminator using train_on_batch(real_images, ones) and train_on_batch(fake_images, zeros)
    # Both loss and accuracy are returned
    # Calculate the mean loss and mean accuracy
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_img, ones)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_img, zeros)
    d_loss = (d_loss_real + d_loss_fake) / 2.0
    d_acc = (d_acc_real + d_acc_fake) / 2.0
    """Train the generator"""
    # Create a random normalized noise
    # Get the loss using train_on_batch(noise, ones) on the combined_model
    noise = np.random.randn(batch_size, latent_space)
    g_loss = combined_model.train_on_batch(noise, ones)
    # Repeat the train process
    noise = np.random.randn(batch_size, latent_space)
    g_loss = combined_model.train_on_batch(noise, ones)
    # Append the losses to both discriminator loss list and generator loss list
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    # Print the progress
    if epoch % 100 == 0:
        print(f"epoch: {epoch + 1}/{epochs}, d_loss: {d_loss:.2f}, \
              d_acc: {d_acc:.2f}, g_loss: {g_loss[0]:.2f}")

    if epoch % sample_period == 0:
        sample_images(epoch)

# Plot the losses
plt.plot(g_losses, label='g_losses')
plt.plot(d_losses, label='d_losses')
plt.legend()
plt.show()
# Show the images
a = imread('gan_images/0.png')
plt.imshow(a)
plt.show()
a = imread('gan_images/10000.png')
plt.imshow(a)
plt.show()
a = imread('gan_images/20000.png')
plt.imshow(a)
plt.show()
a = imread('gan_images/29800.png')
plt.imshow(a)
plt.show()