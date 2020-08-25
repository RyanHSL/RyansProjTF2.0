from tensorflow.keras import layers, models, optimizers, datasets
from PIL import Image

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.random.set_seed(22)
np.random.seed(22)

# Config
new_img = Image.new("L", (280, 280))
image_size = 28*28
h_dim = 512
z_dim = 20
num_epochs = 55
batch_size = 100
learning_rate = 1e-3

def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test

class VAE(models.Model):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = layers.Dense(h_dim)
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)
        self.fc4 = layers.Dense(h_dim)
        self.fc5 = layers.Dense(image_size)

        return

    def encode(self, x):
        h = tf.nn.relu(self.fc1(x))

        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_ver):
        std = tf.exp(log_ver * 0.5)
        eps = tf.random.normal(std.shape)

        return mu + eps * std

    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        x_reconstructed_logits = self.decode_logits(z)

        return x_reconstructed_logits, mu, log_var

def main():
    x_train, y_train, x_test, y_test = get_data()
    model = VAE()
    model.build(input_shape=(4, image_size))
    model.summary()
    optimizer = optimizers.Adam(learning_rate)

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(batch_size * 5).batch(batch_size)
    num_batches = x_train.shape[0] // batch_size

    for epoch in range(num_epochs):
        for step, x in enumerate(dataset):
            x = np.reshape(x, [-1, image_size])

            with tf.GradientTape() as tape:
                x_reconstructed_logits, mu, log_var = model(x)

                reconstructed_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_reconstructed_logits)
                reconstructed_loss = tf.reduce_sum(reconstructed_loss) / batch_size

                kl_div = -0.5 * tf.reduce_sum(1. + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
                kl_div = tf.reduce_mean(kl_div)

                loss = tf.reduce_mean(reconstructed_loss) + kl_div

            gradients = tape.gradient(loss, model.trainable_variables)
            for g in gradients:
                tf.clip_by_norm(g, 15)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if (step + 1) % 50 == 0:
                print(f"Epoch[{epoch + 1}/{num_epochs}], Step[{step + 1}/{num_batches}], "
                      f"Reconst Loss: {float(reconstructed_loss)}, KL Div: {float(kl_div)}")

        # Generator
        z = tf.random.normal((batch_size, z_dim))
        out = model.decode(z)
        out = tf.reshape(out, [-1, 28, 28]).numpy() * 255
        out = out.astype(np.uint8)

        index = 0
        for i in range(0, 280, 28):
            for j in range(0, 280, 28):
                im = out[index]
                im = Image.fromarray(im, mode="L")
                new_img.paste(im, (i, j))
                index += 1

        new_img.save(f"Images/VAE/vae_sampled_epoch_{epoch + 1}.png")
        plt.imshow(np.asarray(new_img))
        plt.show()

        # Save the reconstructed images pf last batch
        out_logits, _, _ = model(x[:batch_size // 2])
        out = tf.nn.sigmoid(out_logits)
        out = tf.reshape(out, [-1, 28, 28]).numpy() * 255

        x = tf.reshape(x[: batch_size // 2], [-1, 28, 28])

        x_concat = tf.concat([x, out], axis=0).numpy() * 255
        x_concat = x_concat.astype(np.uint8)

        index = 0
        for i in range(0, 280, 28):
            for j in range(0, 280, 28):
                im = x_concat[index]
                im = Image.fromarray(im, mode="L")
                new_img.paste(im, (i, j))
                index += 1

        new_img.save(f"Images/VAE/vae_reconstructed_epoch_{epoch + 1}.png")
        plt.imshow(np.asarray(new_img))
        plt.show()
        print("New images saved.")

    return

if __name__ == "__main__":
    main()