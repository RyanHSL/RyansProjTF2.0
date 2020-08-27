from tensorflow.keras import models, layers, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionInput
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vggInput
from glob import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Config
image_size = [100, 100]
batch_size = 128
path = "Data/flowers"


class PretrainedInception(models.Model):

    def __init__(self, input_shape, K):
        super(PretrainedInception, self).__init__()
        ptm = InceptionV3(input_shape=input_shape,
                          include_top=False,
                          weights="imagenet")
        ptm.trainable = False

        x = layers.Flatten()(ptm.output)
        x = layers.Dense(64, activation=tf.nn.relu)(x)
        x = layers.Dense(K, activation=tf.nn.softmax)(x)

        self.model = models.Model(ptm.input, x)

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

class PretrainedVGG(models.Model):

    def __init__(self, input_shape, K):
        super(PretrainedVGG, self).__init__()
        ptm = VGG16(input_shape=input_shape,
                    include_top=False,
                    weights="imagenet")
        ptm.trainable = True

        x = layers.Flatten()(ptm.output)
        x = layers.Dense(64, activation=tf.nn.relu)(x)
        x = layers.Dense(K, activation=tf.nn.softmax)(x)

        self.model = models.Model(ptm.input, x)

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

def get_steps():
    train_step = int(np.ceil(len(glob(path + "/*/*.jpg")) * 0.8)) / batch_size
    val_step = int(np.ceil(len(glob(path + "/*/*.jpg")) * 0.2)) / batch_size

    return train_step, val_step

def generate_data():
    # data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    #                                     zoom_range=2.0, horizontal_flip=True, vertical_flip=True,
    #                                     preprocessing_function=inceptionInput, validation_split=0.2)
    data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                        zoom_range=2.0, horizontal_flip=True, vertical_flip=True,
                                        preprocessing_function=vggInput, validation_split=0.2)
    train_generator = data_generator.flow_from_directory(path, target_size=image_size, class_mode="sparse",
                                                         batch_size=batch_size, shuffle=True, subset="training")
    val_generator = data_generator.flow_from_directory(path, target_size=image_size, class_mode="sparse",
                                                       batch_size=batch_size, shuffle=True, subset="validation")

    return train_generator, val_generator

def main():
    train_generator, val_generator = generate_data()
    x_train, y_train = next(train_generator)
    K = len(set(y_train))
    print(x_train.shape, y_train.shape)
    train_step, val_step = get_steps()

    # model = PretrainedInception(x_train[0].shape, K) # loss: 1.0677 - accuracy: 0.5713 - val_loss: 1.0950 - val_accuracy: 0.5691
    model = PretrainedVGG(x_train[0].shape, K) # loss: 0.7685 - accuracy: 0.7100 - val_loss: 0.8781 - val_accuracy: 0.6539
    model.compile(optimizer=optimizers.Adam(lr=1e-3),
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])
    r = model.fit(train_generator, validation_data=val_generator, epochs=50,
                  steps_per_epoch=train_step, validation_steps=val_step)

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

    plt.plot(r.history["accuracy"], label="accuracy")
    plt.plot(r.history["val_accuracy"], label="val accuracy")
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    main()

