from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

import os
import requests
import shutil
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_path = "seg_train/seg_train"
valid_path = "seg_test/seg_test"
pred_path = "seg_pred/seg_pred"
IMAGE_SIZE = [200, 200]
batch_size = 128

class PretrainedModel(Model):
    def __init__(self, input_shape, K):
        super(PretrainedModel, self).__init__()
        self.K = K
        self.ptm = VGG16(input_shape=input_shape,
                         weights="imagenet",
                         include_top=False)
        self.ptm.trainable = False
        x = Flatten()(self.ptm.output)
        x = Dense(32, activation=tf.nn.relu)(x)
        x = Dense(self.K, activation=tf.nn.softmax)(x)
        model = Model(self.ptm.input, x)

        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

def get_data():
    # os.chdir("Data")
    # r = requests.get("https://www.kaggle.com/puneet6060/intel-image-classification/download", stream=True)
    # if r.status_code == 200:
    #     with open("intel-image-classification.zip", "wb") as f:
    #         r.raw.decode_content = True
    #         shutil.copyfileobj(r.raw, f)
    #
    # with zipfile.ZipFile("intel-image-classification.zip", "r") as zipRef:
    #     zipRef.extractall()
    #
    os.chdir("Data/intel-image-classification")
    train_data = glob(train_path + "/*/*.jpg")
    test_data = glob(valid_path + "/*/*.jpg")
    pred_data = glob(pred_path + "/*.jpg")
    folders = glob(train_path + "/*")
    k = len(folders)

    return train_data, test_data, pred_data, k

def gen_data():
    gen_train = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=preprocess_input)
    gen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
    gen_pred = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = gen_train.flow_from_directory(train_path,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=batch_size)
    test_generator = gen_test.flow_from_directory(valid_path,
                                                  target_size=IMAGE_SIZE,
                                                  batch_size=batch_size)
    pred_generator = gen_pred.flow_from_directory(pred_path,
                                                  target_size=IMAGE_SIZE,
                                                  batch_size=batch_size)

    return train_generator, test_generator, pred_generator

def gen_data_wo_data_augmentation():
    gen_train = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = gen_train.flow_from_directory(train_path,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=batch_size)

    return train_generator

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
    train_images, test_images, pred_images, K = get_data()
    plt.imshow(image.load_img(np.random.choice(train_images)))
    plt.show()
    train_gen, test_gen, pred_gen = gen_data()
    train_gen2 = gen_data_wo_data_augmentation()
    classes = train_gen.class_indices
    print(classes)
    classes = list(classes.keys())

    model = PretrainedModel(IMAGE_SIZE + [3], K)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
    r = model.fit(train_gen,
                  validation_data=test_gen,
                  epochs=10,
                  steps_per_epoch=int(np.ceil(len(train_images)/batch_size)),
                  validation_steps=int(np.ceil(len(test_images)/batch_size)))
    plot_results(r)

    r = model.evaluate(train_gen2, steps=int(np.ceil(len(train_images)/batch_size))) # Evaluate the model without data augmentation

    # predictions = model.predict(pred_gen)
    #
    # for i in range(10):
    #     print(f"The prediction for image {i} is ", classes(np.argmax(predictions[i])))
    #     plt.imshow(image.load_img(pred_images[i]))
    #     plt.show()

    return

if __name__ == "__main__":
    main()