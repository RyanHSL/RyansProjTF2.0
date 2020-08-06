from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The dictionary maybe in different order
# genderDict = {"female": 0, "male": 1}
# raceDict = {"group B": 0, "group C": 1, "group A": 2, "group D": 3, "group E": 4}
# eduDict = {"bachelor's degree": 0, "some college": 1, "master's degree": 2, "associate's degree": 3,
#            "associate's degree": 4, "high school": 5, "some high school": 6}
# prepareDict = {"none": 0, "completed": 1}

class PerformanceModel(Model):
    def __init__(self, gshape, rshape, eshape, wshape, V, D, K):
        super(PerformanceModel, self).__init__()
        self.V = V
        self.D = D
        self.K = K

        g = Input(shape=gshape)
        gEmb = Embedding(self.V, self.D)(g)
        gFlt = Flatten()(gEmb)

        r = Input(shape=rshape)
        rEmb = Embedding(self.V, self.D)(r)
        rFlt = Flatten()(rEmb)

        e = Input(shape=eshape)
        eEmb = Embedding(self.V, self.D)(e)
        eFlt = Flatten()(eEmb)

        p = Input(shape=wshape)
        pEmb = Embedding(self.V, self.D)(p)
        pFlt = Flatten()(pEmb)

        x = Concatenate()([gFlt, rFlt, eFlt, pFlt])
        x = Dense(512, activation=tf.nn.relu)(x)
        x = Dense(1024, activation=tf.nn.relu)(x)
        x = Dense(self.K)(x)

        model = Model(inputs=[g, r, e, p], outputs=x)
        self.model = model

        return

    def call(self, x, training=None):
        m = self.model(x, training=training)

        return m

def get_data():
    df = pd.read_csv("Data/StudentsPerformance.csv")

    df["gender"] = pd.Categorical(df["gender"])
    df["c_gender"] = df["gender"].cat.codes
    df["race/ethnicity"] = pd.Categorical(df["race/ethnicity"])
    df["c_race"] = df["race/ethnicity"].cat.codes
    df["parental level of education"] = pd.Categorical(df["parental level of education"])
    df["c_edu"] = df["parental level of education"].cat.codes
    df["test preparation course"] = pd.Categorical(df["test preparation course"])
    df["c_prep"] = df["test preparation course"].cat.codes
    df = df.drop(["lunch", "gender", "race/ethnicity", "parental level of education", "test preparation course"], axis=1)
    print(df.head())

    math_score, reading_score, writing_score = df["math score"].values, df["reading score"].values, df["writing score"].values
    gender, race, education, preparation = df["c_gender"].values, df["c_race"], df["c_edu"].values, df["c_prep"].values
    # math_score -= tf.reduce_mean(math_score)
    # reading_score -= tf.reduce_mean(reading_score)
    # writing_score -= tf.reduce_mean(writing_score)
    math_score = math_score / 100
    reading_score = reading_score / 100
    writing_score = writing_score / 100

    return gender, race, education, preparation, math_score, reading_score, writing_score

def main():
    gender, race, education, preparation, math_score, reading_score, writing_score = get_data()
    X = np.transpose(np.array([gender, race, education, preparation]))
    Y = np.transpose(np.array([math_score, reading_score, writing_score]))
    V = len(X)
    D = 20 # Embedding dimensionality
    print(X.shape, Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    g_train, r_train, e_train, p_train = np.transpose(X_train)[0], np.transpose(X_train)[1], np.transpose(X_train)[2], np.transpose(X_train)[3]
    g_test, r_test, e_test, p_test = np.transpose(X_test)[0], np.transpose(X_test)[1], np.transpose(X_test)[2], np.transpose(X_test)[3]
    m_train, r_train, w_train = np.transpose(Y_train)[0], np.transpose(Y_train)[1], np.transpose(Y_train)[2]
    m_test, r_test, w_test = np.transpose(Y_test)[0], np.transpose(Y_test)[1], np.transpose(Y_test)[2]

    model = PerformanceModel(g_train[0].shape, r_train[0].shape, e_train[0].shape, w_train[0].shape, V, D, Y_test.shape[1])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-3, momentum=9e-1),
                  loss=tf.keras.losses.mse)

    def schedule(epoch, lr):
        if epoch >= 50:
            return 5e-4
        return 1e-3

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    r = model.fit([g_train, r_train, e_train, p_train], Y_train,
                  validation_data=([g_test, r_test, e_test, p_test], Y_test),
                  epochs=200,
                  batch_size=128,
                  callbacks=[scheduler])

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    main()