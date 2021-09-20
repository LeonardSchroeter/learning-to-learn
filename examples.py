import tensorflow as tf
from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = keras.layers.Dense(32, activation="relu")
        self.linear_2 = keras.layers.Dense(32, activation="relu")
        self.linear_3 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return self.linear_3(x)
