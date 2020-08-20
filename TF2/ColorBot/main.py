from tensorflow import keras
from model import RNNColorBot
from utils import load_dataset, parse

import os, six, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(22)
np.random.seed(22)

SOURCE_TRAIN_URL = "https://raw.githubusercontent.com/random-forests/tensorflow-workshop/master/archive/extras/colorbot/data/train.csv"
SOURCE_TEST_URL = "https://raw.githubusercontent.com/random-forests/tensorflow-workshop/master/archive/extras/colorbot/data/test.csv"


def validate(model, eval_data):
    avg_loss = keras.metrics.Mean()

    for (labels, chars, sequence_length) in eval_data:
        predictions = model((chars, sequence_length), training=False)
        avg_loss.update_state((keras.losses.mean_squared_error(labels, predictions)))

    print(f"eval/loss:{avg_loss.result().numpy()}")

    return

def train_one_step(model, optimizer, train_data, log_interval, epoch):
    for step, (labels, chars, sequence_length) in enumerate(train_data):
        with tf.GradientTape() as tape:
            predictions = model((chars, sequence_length), training=True)
            loss = keras.losses.mean_squared_error(labels, predictions)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, "loss:", float(loss))

def main():
    batch_size = 64
    rnn_cell_size = [256, 128]
    epochs = 40

    data_dir = os.path.join(".", "Data")
    # data_dir = "Data"
    train_data = load_dataset(data_dir, SOURCE_TRAIN_URL, batch_size)
    test_data = load_dataset(data_dir, SOURCE_TEST_URL, batch_size)

    model = RNNColorBot(rnn_cell_size, 3, 0.5)
    optimizer = keras.optimizers.Adam(1e-3)

    for epoch in range(epochs):
        start = time.time()
        train_one_step(model, optimizer, train_data, 50, epoch)
        end = time.time()

        if epochs % 10 == 0:
            validate(model, test_data)

    print("ColorBot is ready to generate colors.")
    while True:
        try:
            color_name = six.moves.input("Give me a color name (or press enter to exit): ")
        except EOFError:
            return

        if not color_name:
            return

        _, chars, length = parse(color_name)

        (chars, length) = (tf.identity(chars), tf.identity(length))
        chars = tf.expand_dims(chars, 0)
        length = tf.expand_dims(length, 0)
        preds = tf.unstack(model((chars, length), training=False)[0])
        clipped_preds = tuple(min(float(p), 1.0) for p in preds)
        rgb = tuple(int(p*255) for p in clipped_preds)
        print("rgb", rgb)
        data = [[clipped_preds]]

        plt.imshow(data)
        plt.title(color_name)
        plt.savefig(color_name + ".png")

    return

if __name__ == "__main__":
    main()
