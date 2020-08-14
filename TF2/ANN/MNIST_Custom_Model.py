from tensorflow.keras.models import Model
from tensorflow.keras import layers, optimizers, datasets
from tensorflow import keras

import tensorflow as tf
import numpy as np

def prepare_data(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y

def get_data():
    (x_train, y_train), (x_val, y_val) = datasets.mnist.load_data()
    y_train, y_val = tf.one_hot(y_train, depth=10), tf.one_hot(y_val, depth=10)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.map(prepare_data)
    ds_train = ds_train.shuffle(60000).batch(100)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(prepare_data)
    ds_val = ds_val.shuffle(10000).batch(100)

    sample = next(iter(ds_train))
    print("sample: ", sample[0].shape, sample[1].shape)

    return ds_train, ds_val

class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.layer1 = layers.Dense(100, activation=tf.nn.relu)
        self.layer2 = layers.Dense(50, activation=tf.nn.relu)
        self.layer3 = layers.Dense(10)

        return

    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 28*28])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

def main():
    tf.random.set_seed(22)

    train_dataset, val_dataset = get_data()
    model = MyModel()
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    model.fit(train_dataset.repeat(), epochs=20, steps_per_epoch=500, verbose=1,
              validation_data=val_dataset.repeat(), validation_steps=2)

    return

if __name__ == "__main__":
    main()