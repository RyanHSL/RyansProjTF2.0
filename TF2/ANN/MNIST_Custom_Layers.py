from tensorflow.keras import layers, optimizers, datasets

import tensorflow as tf
import numpy as np

def prepare_data(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y

def get_data():
    (x, y), _ = datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_data)
    ds = ds.take(20000).shuffle(20000).batch(100)

    return ds

def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

def compute_acc(logits, labels):
    predictions = tf.argmax(logits, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_acc(logits, y)

    return loss, accuracy

def train(epoch, model, optimizer):
    train_ds = get_data()
    loss = accuracy = 0.0

    for step, (x, y) in enumerate(train_ds):
        loss, accuracy = train_one_step(model, optimizer, x, y)

        if step % 500 == 0:
            print(f"epoch{epoch}: loss{loss.numpy()}; accuracy{accuracy.numpy()}")

    return loss, accuracy

class MyLayer(layers.Layer):

    def __init__(self, units):
        super(MyLayer, self).__init__()

        for i in range(1, len(units)):
            self.add_variable(name="kernel%d"%i, shape=[units[i - 1], units[i]])
            self.add_variable(name="bias%d"%i, shape=[units[i]])

        return

    def call(self, x):
        num = len(self.trainable_variables)
        x = tf.reshape(x, [-1, 28*28])

        for i in range(0, num, 2):
            x = tf.matmul(x, self.trainable_variables[i]) + self.trainable_variables[i + 1]

        return x

def main():
    model = MyLayer([28*28, 200, 200, 10])

    for v in model.trainable_variables:
        print(v.name, v.shape)

    optimizer = optimizers.Adam()

    for epoch in range(20):
        loss, accuracy = train(epoch, model, optimizer)

    print(f"Final epoch{epoch}: loss{loss.numpy()}; accuracy{accuracy.numpy()}")

    return

if __name__ == "__main__":
    main()