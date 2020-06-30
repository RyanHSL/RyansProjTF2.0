from tensorflow.keras.layers import Input, SimpleRNN, Dense, GlobalAvgPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, tanh

class GRU(Model):
    def _init_(self, input_shape=(10, 1), m=15, activation=relu):
        super(GRU, self)._init_()

        self.input_shape=input_shape

        i = Input(input_shape=self.input_shape)
        x = GRU(m, activation=activation)(i)
        x = Dense(1, activation=sigmoid)(x)
        model = Model(i, x)
        self.model = model

    def call(self):
        return self.model()