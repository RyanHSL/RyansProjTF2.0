from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid

class Linear(Model):
    def _init_(self, input_shape=(1), activation=sigmoid):
        super(Linear, self)._init_()

        self.input_shape = input_shape

        i = Input(input_shape=self.input_shape)
        x = Dense(1, activation=activation)(i)
        model = Model(i, x)
        self.model = model

    def call(self, x):
        x = self.model(x)
        return x