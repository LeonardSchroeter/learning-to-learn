import tensorflow as tf
from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = keras.layers.Linear(32)
        self.linear_2 = keras.layers.Linear(32)
        self.linear_3 = keras.layers.Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)
