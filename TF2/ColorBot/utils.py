from tensorflow import keras

import os
import tensorflow as tf
import numpy as np
import urllib

def parse(line):

    items = tf.strings.split([line], ",").values # color_name, r, g, b -> [color_name, r, g, b]
    rgb = tf.strings.to_number(items[1:], out_type=tf.float32) / 255.
    # Represent the color name as a one-hot encoded character sequence.
    color_name = items[0]
    chars = tf.one_hot(tf.io.decode_raw(color_name, tf.uint8), depth=256)
    # The sequence length is needed by our RNN.
    # length = tf.cast(chars.shape[0], dtype=tf.int64)
    length = tf.cast(tf.shape(chars)[0], dtype=tf.int64)


    return rgb, chars, length

def download(fileName, work_directory, source_url):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)

    filePath = os.path.join(work_directory, fileName)

    if not os.path.exists(filePath):
        temp_file_name, _ = urllib.request.urletrieve(source_url)
        tf.io.gfile.copy(temp_file_name, filePath)

        with tf.io.gfile.GFile(filePath) as f:
            size = f.size()
            print("Successfully downloaded", fileName, size, "bytes.")

    return filePath

def load_dataset(data_dir, url, batch_size):
    path = download(os.path.basename(url), data_dir, url)
    dataset = tf.data.TextLineDataset(path).skip(1).map(parse).shuffle(
                buffer_size=10000).padded_batch(
                batch_size, padded_shapes=([None], [None, None], []))

    return dataset