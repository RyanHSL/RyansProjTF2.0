from tensorflow.keras.layers import Input, Conv1D, LSTM, Embedding, Flatten, Dropout, Dense, GlobalMaxPooling1D, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from time import sleep

import requests
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import tempfile

def show(idx, category):
    print("%s"%category, X_test[idx])

if __name__ == "__main__":
    # Preprocess the data
    df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

    # Build a RNN spam detection model
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df.columns = ["Labels", "Data"]
    df["b_labels"] = df["Labels"].map({"ham": 0, "spam": 1})
    Y = df["b_labels"].values
    X = df["Data"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    MAX_VOCAB = 20000
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(X_train)
    train_sequence = tokenizer.texts_to_sequences(X_train)
    test_sequence = tokenizer.texts_to_sequences(X_test)

    word2Index = tokenizer.word_index
    V = len(word2Index) + 1
    train_data = pad_sequences(train_sequence)
    T = train_data.shape[1]
    test_data = pad_sequences(test_sequence, maxlen=T)

    D, M = 20, 15

    i = Input(shape=(T, ))
    x = Embedding(V, D)(i)
    x = LSTM(M, activation="relu", return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation="softmax")(x)
    model = Model(i, x)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    # Save the model to a temporary directory
    MODEL_DIR = tempfile.gettempdir()
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print("export_path = {}\n".format(export_path))
    if os.path.exists(export_path):
        print("\nAlready saved a model, cleaning up\n")
        shutil.rmtree(export_path)

    # Save the model
    tf.saved_model.save(model, export_path)

    os.environ["MODEL_DIR"] = MODEL_DIR

    # %%bash --bg
    # nohup tensorflow_model_server \
    #   --rest_api_port=8501 \
    #   --model_name=spam_detection_model \
    #   --model_base_path="${MODEL_DIR}" >server.log 2>&1

    # Label mapping
    labels = """Ham
    Spam""".split("\n")

    # Print the result
    i = np.random.randint(0, len(test_data))
    show(i, labels[Y_test[i]])

    data = json.dumps({"signature_name": "serving_default", "instance": test_data[0:10].tolist()})
    print(data)

    headers = {"content-type": "application/json"}
    r = requests.post("http://8501/v1/spam_detection_model:predict", data=data, headers=headers)
    try:
        j = r.json()
        sleep(1)
        print(j.keys())
        print(j)

        pred = np.array(j["predictions"])
        print(pred.shape)

        # Get the predicted classes
        pred = pred.argmax(axis=1)

        # Map them back to strings
        pred = [labels[i] for i in pred]
        print(pred)

        # Get the true classification
        actual = [labels[i] for i in Y_test[0:10]]
        print(actual)

        for i in range(0, 10):
            show(i, f"True: {actual[i]}, Predicted: {[pred[i]]}")
    except requests.exceptions.ConnectionError:
        r.status_code = "Connection refused"

    # Allow me to select a model by version
    headers = {"content-type": "application/json"}
    r = requests.post("http://localhost:8501/v1/models/spam_detection_model/versions/1:predict", data=data, headers=headers)
    j = r.json()
    pred = np.array(j["predictions"])
    pred = pred.argmax(axis=1)
    pred = [labels[i] for i in pred]
    for i in range(0, 10):
        show(i, f"True: {actual[i]}, Predicted: {[pred[i]]}")

    # Build a second model
    i = Input(shape=(T, ))
    x = Embedding(T, D)(i)
    x = Conv1D(32, 3, activation="relu")(x)
    x = MaxPool1D(2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(64, 3, activation="relu")(x)
    x = MaxPool1D(2)(x)
    x = Flatten()(x)
    x = Dense(1, activation="softmax")(x)
    model2 = Model(i, x)

    model2.compile(optimzer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    version = 2
    export_path = os.path.join(MODEL_DIR, str(version))
    print("export_path = {}\n".format(export_path))
    if os.path.isdir(export_path):
        print("\nAlready saved a model, cleaning up\n")
        shutil.rmtree(export_path)

    tf.saved_model.save(model2, export_path)
    print("\nSaved model:")

    headers = {"content-type": "application/json"}
    r = requests.post("http://localhost:8051/v1/models/spam_detection/version/2:predict", data=data, headers=headers)

    try:
        j = r.json()
        sleep(1)
        pred = np.array(j["predictions"])
        pred = pred.argmax(axis=1)
        pred = [labels[i] for i in pred]
        for i in range(0, 10):
            show(1, f"True: {actual[i]}, Predicted: {pred[i]}")
    except requests.exceptions.ConnectionError:
        r.status_code = "Connection refused"

