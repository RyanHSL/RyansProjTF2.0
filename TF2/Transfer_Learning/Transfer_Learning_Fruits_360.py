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
test_path = "seg_test/seg_test"
pred_path = "seg_pred/seg_pred"
batch_size = 128
IMAGE_SIZE = [200, 200]

def get_data():
    # os.chdir("Data")
    # r = requests.get("https://www.kaggle.com/puneet6060/intel-image-classification/download", stream=True)
    # if r.status_code == 200:
    #     with open("intel-image-classification.zip", "wb") as f:
    #         r.raw.decode_content = True
    #         shutil.copyfileobj(r.raw, f)
    # with zipfile.ZipFile("intel-image-classification.zip", "r") as zipRef:
    #     zipRef.extractall()
    #
    # os.chdir("intel-image-classification")
    # It requires the Kaggle Credential to download the data. The zip file is damaged after running the above code, so please manually download it.
    os.chdir("Data/intel-image-classification")

    train_data = glob(train_path + "/*/*.jpg")
    valid_data = glob(test_path + "/*/*.jpg")
    pred_data = glob(pred_path + "/*.jpg")
    folders = glob(train_path + "/*")
    K = len(folders)

    return train_data, valid_data, pred_data, K

class PretrainedModel(Model):
    def __init__(self, input_shape, K):
        super(PretrainedModel, self).__init__()
        self.K = K
        self.ptm = VGG16(input_shape=input_shape,
                         weights="imagenet",
                         include_top=False)
        self.ptm.trainable = False
        x = Flatten()(self.ptm.output)
        x = Dense(self.K, activation=tf.nn.softmax)(x)
        model = Model(self.ptm.input, x)

        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

def gen_data():
    gen_train = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=preprocess_input)
    gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = gen_train.flow_from_directory(train_path,
                                                    target_size=IMAGE_SIZE,
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    class_mode="categorical")
    val_generator = gen_val.flow_from_directory(test_path,
                                                target_size=IMAGE_SIZE,
                                                batch_size=batch_size)

    return train_generator, val_generator


def main():
    train_data, valid_data, pred_data, K = get_data()
    plt.imshow(image.load_img(np.random.choice(train_data)))
    plt.show()
    train_gen, val_gen = gen_data()
    model = PretrainedModel(IMAGE_SIZE + [3], K)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    r = model.fit(train_gen,
                  validation_data=val_gen,
                  epochs=10,
                  steps_per_epoch=int(np.ceil(len(train_data)/batch_size)),
                  validation_steps=int(np.ceil(len(valid_data)/batch_size)))
    # loss: 2.5937 - accuracy: 0.8879 - val_loss: 3.4541 - val_accuracy: 0.8917
    model.evaluate_generator(train_gen, steps=int(np.ceil(len(train_data)/batch_size)))

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