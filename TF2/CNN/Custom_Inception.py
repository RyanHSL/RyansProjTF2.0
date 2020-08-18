from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from glob import glob

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
import shutil
import zipfile


class ConvBNRelu(keras.Model):

    def __init__(self, ch, kernal_size=3, strides=1, padding="same"):
        super(ConvBNRelu, self).__init__()

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(ch, kernal_size, strides=strides, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

        return

    def call(self, x, training=None):
        x = self.model(x, training=training)

        return x

class InceptionBlk(keras.Model):

    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()

        self.ch = ch
        self.strides = strides

        self.conv1 = ConvBNRelu(ch, strides=strides)
        self.conv2 = ConvBNRelu(ch, kernal_size=3, strides=strides)
        self.conv3_1 = ConvBNRelu(ch, kernal_size=3, strides=strides)
        self.conv3_2 = ConvBNRelu(ch, kernal_size=3, strides=1)

        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding="same")
        self.pool_conv = ConvBNRelu(ch, strides=strides)

        return

    def call(self, x, training=None):

        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)

        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)

        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)

        x = tf.concat([x1, x2, x3_2, x4], axis=3)

        return x

class Inception(keras.Model):

    def __init__(self, num_layers, num_classes, init_ch=16, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_layers = num_layers
        self.init_ch = init_ch

        self.conv1 = ConvBNRelu(init_ch)

        self.blocks = keras.models.Sequential(name="dynamic-blocks")

        for block_id in range(num_layers):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)

                self.blocks.add(block)

            self.out_channels *= 2

        self.avg_pool = keras.layers.GlobalMaxPooling2D()
        self.fc = keras.layers.Dense(num_classes)
        # self.fa = keras.layers.Activation("softmax")

        return

    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.blocks(out,training=training)
        out = self.avg_pool(out)
        out = self.fc(out)
        # out = self.fa(out)

        return out

def get_data(path):
    # os.chdir("Data")
    # r = requests.get("https://www.kaggle.com/alxmamaev/flowers-recognition", stream=True)
    # if r.status_code == 200:
    #     with open("23777_30378_bundle_archive.zip", "wb") as f:
    #         r.raw.decode_content = True
    #         shutil.copyfileobj(r.raw, f)
    #
    # with zipfile.ZipFile("23777_30378_bundle_archive.zip", "r") as zipRef:
    #     zipRef.extractall()

    data = glob(path + "/*/*jpg")
    num_classes = len(glob(path + "/*"))

    return data, num_classes

def generate_data(target_size, path):
    gen_data = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, rotation_range=20, zoom_range=2.0,
                                  horizontal_flip=True, vertical_flip=True, validation_split=0.2, rescale=1.0/255.0)
    train_data = gen_data.flow_from_directory(path,
                                              shuffle=True,
                                              target_size=target_size,
                                              class_mode="categorical",
                                              subset="training")
    val_data = gen_data.flow_from_directory(path,
                                            target_size=target_size,
                                            class_mode="categorical",
                                            subset="validation")

    return train_data, val_data

def plot_results(r):
    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()
    plt.plot(r.history["accuracy"], label="accuracy")
    plt.plot(r.history["val_accuracy"], label="val accuracy")
    plt.legend()
    plt.show()

def main():
    batch_size = 256
    target_size = [100, 100]
    epochs = 100
    path = "Data/flowers"

    data, K = get_data(path)
    train_data, val_data = generate_data(target_size, path)
    model = Inception(2, K)
    # inupt_shape = target_size + [3]
    model.build(input_shape=(None, 100, 100, 3))
    model.summary()

    optimizer = keras.optimizers.Adam(lr=1e-3)
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    acc_meter = keras.metrics.Accuracy()

    # model.compile(optimizer=optimizer,
    #               loss=keras.losses.categorical_crossentropy,
    #               metrics=["accuracy"])
    # r = model.fit(train_data, validation_data=val_data, epochs=epochs,
    #               steps_per_epoch=int(np.ceil(len(train_data)/batch_size)),
    #               validation_steps=int(np.ceil(len(val_data)/batch_size)))
    #
    # plot_results(r)
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print(epoch, step, "loss:", loss.numpy())

        acc_meter.reset_states()
        for x, y in enumerate(val_data):
            logits = model(x, training=False)
            pred = tf.argmax(logits, axis=1)
            acc_meter.update_state(y, pred)

        print(epoch, "evaluation acc:", acc_meter.result().numpy())

    return

if __name__ == "__main__":
    main()
