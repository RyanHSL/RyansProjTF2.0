from tensorflow.keras import datasets, Model, layers, optimizers
from PIL import Image

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.random.set_seed(22)
np.random.seed(22)
# Config
new_im = Image.new("L", (280, 280))

image_size = 28*28
h_dim = 20
num_epochs = 55
batch_size = 128
learning_rate = 1e-3

def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = np.float32(x_train) / 255.0, np.float32(x_test) / 255.0

    return x_train, y_train, x_test, y_test

class AE(Model):

    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(h_dim)
        self.fc3 = layers.Dense(512)
        self.fc4 = layers.Dense(image_size)

        return

    def encode(self, x):
        x = tf.nn.relu(self.fc1(x))
        x = (self.fc2(x))

        return x

    def decode_logits(self, h):
        x = tf.nn.relu(self.fc3(h))
        x = self.fc4(x)

        return x

    def decode(self, h):
        return tf.nn.sigmoid(self.decode_logits(h))

    def call(self, inputs, training=None, mask=None):
        h = self.encode(inputs)
        x_reconstructed_logits = self.decode_logits(h)

        return x_reconstructed_logits

def main():
    model = AE()
    model.build(input_shape=(4, image_size))
    model.summary()
    optimizer = optimizers.Adam(learning_rate)

    x_train, y_train, x_test, y_test = get_data()
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(batch_size * 5).batch(batch_size)

    num_batches = x_train.shape[0] // batch_size

    for epoch in range(num_epochs):
        for step, x in enumerate(dataset):
            x = tf.reshape(x, [-1, image_size])
            with tf.GradientTape() as tape:
                x_reconstruction_logits = model(x)
                reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(x, x_reconstruction_logits)
                reconstruction_loss = tf.reduce_sum(reconstruction_loss) / batch_size

            gradient = tape.gradient(reconstruction_loss, model.trainable_variables)
            gradient, _ = tf.clip_by_global_norm(gradient, 15)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            if (step + 1) % 50 == 0:
                print(f"Epoch[{epoch + 1}/{num_epochs}], Step [{step + 1}/{num_batches}], Reconst Loss: {float(reconstruction_loss)}")

    out_logits = model(x[:batch_size // 2])
    out = tf.nn.sigmoid(out_logits)
    out = tf.reshape(out, [-1, 28, 28]).numpy() * 255

    x = tf.reshape(x[:batch_size // 2], [-1, 28, 28])

    x_concat = tf.concat([x, out], axis=0).numpy() * 255.0
    x_concat = x_concat.astype(np.uint8)

    index = 0
    for i in range(1, 280, 28):
        for j in range(0, 280, 28):
            im = x_concat[index]
            im = Image.fromarray(im, mode="L")
            new_im.paste(im, (i, j))
            index += 1

    new_im.save("Images/vae_reconstructed_epoch_%d.png" % (epoch + 1))
    plt.imshow(np.asarray(new_im))
    plt.show()
    print("New image saved.")

    return

if __name__ == "__main__":
    main()