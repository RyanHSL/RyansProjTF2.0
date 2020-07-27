from GAN import Generator, Discriminator
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from skimage.io import imread

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def save_results(val_out, val_block_size, image_fn, color_mode):
    def preprocess(img):
        img = ((img + 1.0)).astype(np.uint8)
        return img

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # Concat image into a row
        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis=1)

        # Concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # Reset the single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        # Remove single-dimensional entry
        final_image = np.squeeze(final_image, axis=2)

    Image.fromarray(final_image, mode=color_mode).save(image_fn)

# Shorten sigmoid cross entropy loss calculation
def celoss_ones(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.binary_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.ones_like(logits)*(1.0 - smooth)))

def celoss_zeros(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.zeros_like(logits)*(1.0 - smooth)))

def d_loss_fn(generator, discriminator, input_noise, real_image, is_training):
    fake_image = generator(input_noise, is_training)
    d_real_logits = discriminator(real_image, is_training)
    d_fake_logits = discriminator(fake_image, is_training)

    d_loss_real = celoss_ones(d_real_logits, smooth=0.1)
    d_loss_fake = celoss_zeros(d_fake_logits, smooth=0.1)
    loss = (d_loss_real + d_loss_fake) / 2

    return loss

def g_loss_fn(generator, discriminator, input_noise, is_training):
    fake_image = generator(input_noise, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits, smooth=0.1)

    return loss

def main():
    # Configuraton
    latent_dim = 100
    epochs = 50000
    batch_size = 128
    learning_rate = 2e-4
    is_training = True
    sample_period = 1000

    # Load in the data
    data = fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = data.load_data()

    X_train, X_test = X_train.astype(np.float32) / 255, X_test.astype(np.float32) / 255
    N, W, H = X_train.shape
    D = W * H
    X_train, X_test = X_train.reshape(-1, D), X_test.reshape(-1, D)

    # Mirror Strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        discriminator = Discriminator()
        discriminator(D)
        discriminator.compile(optimizer=Adam(learning_rate, 0.5),
                              loss="binary_crossentropy",
                              metrics=["accuracy"])

        generator = Generator()
        generator = generator(latent_dim, D)
        Z = Input(shape=(latent_dim,))
        image = generator(Z)
        discriminator.trainable = False
        fake_pred = discriminator(image)
        combined_model = Model(Z, fake_pred)
        combined_model.compile(optimizer=Adam(learning_rate, 0.5),
                               loss="binary_crossentropy",
                               metrics=["accuracy"])

    # Fit the model. Note: the fit part is outside of the Mirror Strategy
    ones = np.ones(batch_size)
    zeros = np.zeros(batch_size)
    d_losses = []
    g_losses = []
    if not os.path.exists("gan_images"):
        os.mkdir("gan_images")

    def sample_images(sample_epoch):
        rows, cols = 5, 5
        sample_noise = np.random.randn(rows * cols, latent_dim)
        img = generator.predict(noise)
        img = img/2.0 + 0.5
        fig, axs = plt.subplots(rows, cols)
        sample_idx = 0
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(img[sample_idx].reshape(28, 28), cmap="gray")
                axs[i, j].axis("off")
                sample_idx += 1

        fig.savefig("gan_image/%d" % sample_epoch)
        plt.close()

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_img = X_train[idx]
        noise = np.random.randn(batch_size, latent_dim)
        fake_img = generator.predict(noise)
        # d_loss = d_loss_fn(generator, discriminator, noise, real_img, is_training)
        d_loss_real, d_acc_real = discriminator.train_on_batch(real_img, ones)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_img, zeros)
        d_loss = (d_loss_real + d_loss_fake) / 2.0
        d_acc = (d_acc_real + d_acc_fake) / 2.0
        noise = np.random.randn(batch_size, latent_dim)
        g_loss = combined_model.train_on_batch(noise, ones)
        noise = np.random.randn(batch_size, latent_dim)
        g_loss = combined_model.train_on_batch(noise, ones)
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if epoch % 100 == 0:
            print(f"epoch: {epoch + 1}/{epochs}, d_loss: {d_loss:.2f}, \
                  d_acc: {d_acc:.2f}, g_loss: {g_loss[0]:.2f}")

        if epoch % sample_period == 0:
            sample_images(epoch)

    plt.plot(g_losses, label='g_losses')
    plt.plot(d_losses, label='d_losses')
    plt.legend()
    plt.show()

    a = imread('gan_images/0.png')
    plt.imshow(a)
    plt.show()
    a = imread('gan_images/10000.png')
    plt.imshow(a)
    plt.show()
    a = imread('gan_images/20000.png')
    plt.imshow(a)
    plt.show()
    a = imread('gan_images/30000.png')
    plt.imshow(a)
    plt.show()
    a = imread('gan_images/40000.png')
    plt.imshow(a)
    plt.show()
    a = imread('gan_images/49800.png')
    plt.imshow(a)
    plt.show()

if __name__ == "__main__":
    main()