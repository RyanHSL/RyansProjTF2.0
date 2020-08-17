from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from sklearn.model_selection import train_test_split
from glob import glob
from Custom_VGG import VGG16

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import shutil
import zipfile


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
    #

    data = glob(path + "/*/*.jpg")
    num_classes = len(glob(path + "/*"))

    return data, num_classes

def generate_data(target_size, path):
    gen_data = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, rotation_range=20, zoom_range=2.0,
                                  horizontal_flip=True, vertical_flip=True, validation_split=0.2, rescale=1.0/255.0)
    train_data = gen_data.flow_from_directory(path,
                                              shuffle=True,
                                              class_mode="categorical",
                                              target_size=target_size,
                                              subset="training")
    val_data = gen_data.flow_from_directory(path,
                                            shuffle=True,
                                            target_size=target_size,
                                            class_mode="categorical",
                                            subset="validation")

    return train_data, val_data

def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def main():
    tf.random.set_seed(22)
    # Config
    target_size = [100, 100]
    batch_size = 256
    path = "Data/flowers"

    data, num_classes = get_data(path)
    train_data, val_data = generate_data(target_size, path)
    # train_loader = tf.data.Dataset.from_tensor_slices(train_data)
    # val_loader = tf.data.Dataset.from_tensor_slices(val_data)

    plt.imshow(image.load_img(np.random.choice(data)))
    plt.show()

    model = VGG16(target_size + [3], num_classes)

    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.CategoricalAccuracy()
    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(200):
        for step, (x, y) in enumerate(train_data):
            # y = tf.squeeze(y, axis=1)
            # y = tf.one_hot(y, depth=)
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)
                # loss2 = compute_loss(logits, tf.argmax(y, axis=1))
                # mse_loss = tf.reduce_sum(tf.squeeze(y - logits))
                metric.update_state(y, logits)

            # if epoch > 100:
            #     if epoch > 150:
            #         optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)
            #     else:
            #         optimizer = optimizers.SGD(lr=1e-3, momentum=0.9)
            # else:
            #     optimizer = optimizers.SGD(lr=1e-2, momentum=0.9)

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print(epoch, step, "loss:", float(loss), "acc:", metric.result().numpy())
                metric.reset_states()

        if epoch % 1 == 0:
            metric = keras.metrics.CategoricalCrossentropy()
            for x, y in val_data:
                # y = tf.squeeze(y, axis=1)
                # y = tf.one_hot(y, depth=10)

                logits = model.predict(x)
                metric.update_state((y, logits))

            print('test acc:', metric.result().numpy())
            metric.reset_states()



    return

if __name__ == "__main__":
    main()