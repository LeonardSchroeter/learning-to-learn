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

class ConvNN(keras.layers.Layer):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.layer1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.layer2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.layer3 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.layer4 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.layer5 = keras.layers.Flatten()
        self.layer6 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.layer6(x)
