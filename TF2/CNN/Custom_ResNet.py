from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from glob import glob
from Custom_VGG import VGG16

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import shutil
import zipfile


# Config
TEST_PATH = "Data/cats_dogs/test_set/test_set"
TRAIN_PATH = "Data/cats_dogs/training_set/training_set"
TARGET_SIZE = [32, 32]
BATCH_SIZE = 32

def get_data(path):
    # r = requests.get("https://www.kaggle.com/jessicali9530/stanford-dogs-dataset", stream=True)
    # if r.status_code == 200:
    #     with open("119698_791828_bundle_archive.zip", "wb") as f:
    #         r.raw.decode_content = True
    #         shutil.copyfileobj(r.raw, f)
    #
    # with zipfile.ZipFile("119698_791828_bundle_archive.zip", "r") as zipRef:
    #     zipRef.extractall()

    data = glob(TEST_PATH + "/*/*.jpg")
    num_classes = len(glob(path + "/*"))

    return data, num_classes

def generate_data():
    gen_data = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, rotation_range=20, zoom_range=2.0,
                                  horizontal_flip=True, rescale=1.0/255.0) # validation_split=0.2
    train_data = gen_data.flow_from_directory(TRAIN_PATH,
                                              shuffle=True,
                                              target_size=TARGET_SIZE,
                                              class_mode="categorical",
                                              batch_size=BATCH_SIZE) # subset="training"
    val_data = gen_data.flow_from_directory(TEST_PATH,
                                            shuffle=True,
                                            target_size=TARGET_SIZE,
                                            class_mode="categorical",
                                            batch_size=BATCH_SIZE) # subset="validation"

    return train_data, val_data

def conv3x3(channels, stride=1, kernel=(3, 3)):
    return keras.layers.Conv2D(channels, kernel, strides=stride, padding='same',
                               use_bias=False,
                            kernel_initializer=tf.random_normal_initializer())

class ResnetBlock(keras.Model):

    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels, strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # this module can be added into self.
        # however, module in for can not be added.
        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x


class ResNet(keras.Model):

    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels, strides=2, residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels, residual_path=residual_path)

                self.in_channels = self.out_channels

                self.blocks.add(block)

            self.out_channels *= 2

        self.final_bn = keras.layers.BatchNormalization()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):

        out = self.conv_initial(inputs)

        out = self.blocks(out, training=training)

        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)


        return out

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
    epochs = 200
    _, K = get_data(TRAIN_PATH)
    train_data, val_data = generate_data()

    model = ResNet(TARGET_SIZE + [3], K)
    # model = VGG16(TARGET_SIZE + [3], K) # Uncomment the final activation layer of VGG16 after uncommenting this line
    model.compile(optimizer=keras.optimizers.SGD(1e-3),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    def schedule(epoch, lr):
        if epochs > 100:
            if epochs > 150:
                lr = 1e-4
            else:
                lr = 1e-3
        else:
            lr = 1e-2

        return lr

    scheduler = keras.callbacks.LearningRateScheduler(schedule)

    r = model.fit(train_data,
                  validation_data=val_data,
                  epochs=epochs,
                  callbacks=[scheduler],
                  steps_per_epoch=int(np.ceil(len(train_data) / BATCH_SIZE)),
                  validation_steps=int(np.ceil(len(val_data) / BATCH_SIZE)))

    plot_results(r)

    return

if __name__ == "__main__":
    main()