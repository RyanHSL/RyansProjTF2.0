from tensorflow.keras import layers, models

import tensorflow as tf

class RNNColorBot(models.Model):

    def __init__(self, rnn_cell_sizes, label_dimension, keep_prob):
        super(RNNColorBot, self).__init__(name="")
        self.rnn_cell_sizes = rnn_cell_sizes
        self.label_dimension = label_dimension
        self.keep_prob = keep_prob

        self.cells = [layers.LSTMCell(size) for size in rnn_cell_sizes]
        self.relu = layers.Dense(label_dimension, activation=tf.nn.relu)

        return

    def call(self, inputs, training=None):
        (chars, sequence_length) = inputs
        chars = tf.transpose(chars, [1, 0, 2]) # [time_steps, batch_size, dimension]
        batch_size = int(chars.shape[1])
        for l in range(len(self.cells)):
            cell = self.cells[l]
            outputs = []
            state = (tf.zeros((batch_size, self.rnn_cell_sizes[l])),
                     (tf.zeros((batch_size, self.rnn_cell_sizes[l]))))

            chars = tf.unstack(chars, axis=0)
            for ch in chars:
                output, state = cell(ch, state)
                outputs.append(output)

            chars = tf.stack(outputs, axis=0)
            if training:
                chars = tf.nn.dropout(chars, self.keep_prob)

        batch_range = [i for i in range(batch_size)]
        indices = tf.stack([sequence_length - 1, batch_range], axis=1)
        # [t, b, h]
        hidden_states = tf.gather_nd(chars, indices)

        return self.relu(hidden_states)

