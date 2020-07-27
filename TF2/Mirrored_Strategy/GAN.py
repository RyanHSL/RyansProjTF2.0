from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten, MaxPool2D, Dropout, Dense
from tensorflow.keras.models import Model

import tensorflow as tf

class Generator(Model):

    def __init__(self, M=256):
        super(Generator, self).__init__()

        self.neurons = M

        return

    def call(self, latent_dim, D, training=None):
        i = Input(shape=(latent_dim,))
        x = Dense(self.neurons, activation=LeakyReLU(alpha=0.2))(i)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(self.neurons * 2, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(self.neurons * 4, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(D, activation="tanh")(x)
        model = Model(i, x)

        return model

class Discriminator(Model):

    def __init__(self, M=512, K=1):
        super(Discriminator, self).__init__()

        self.neurons = M
        self.output_feature = K

        return

    def call(self, img_size, training=None):
        i = Input(shape=(img_size,))
        x = Dense(self.neurons, activation="relu")(i)
        x = Dense(self.neurons * 2, activation="relu")(x)
        x = Dense(self.output_feature, activation="sigmoid")(x)
        model = Model(i, x)

        return model