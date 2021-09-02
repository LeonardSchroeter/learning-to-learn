import tensorflow as tf
from tensorflow import keras


class QuadraticFunction(keras.layers.Layer):
    def __init__(self, dimension, **kwargs):
        super(QuadraticFunction, self).__init__(**kwargs)
        self.dimension = dimension
        self.W = tf.random.normal([dimension, dimension])
        self.y = tf.random.normal([dimension])

    def call(self, inputs):
        return tf.norm(tf.linalg.matvec(self.W, inputs) - self.y)
