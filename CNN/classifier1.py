import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np


class Classifier1:
    model = None
    X = None
    y = None

    def start(self):
        self.load_training_data()
        self.make_model()
        self.fit()

    def load_training_data(self):
        self.X = pickle.load(open("X.pickle", "rb"))
        self.y = np.array(pickle.load(open("y.pickle", "rb")))

    def make_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(64, (3, 3), input_shape=self.X.shape[1:]))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))

        self.model.add(Dense(1))
        self.model.add(Activation("sigmoid"))

        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def fit(self):
        self.model.fit(self.X, self.y, batch_size=32, validation_split=0.1, epochs=3)