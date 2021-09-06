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

class QuadraticFunctionLayer(keras.layers.Layer):
    def __init__(self, dimension, **kwargs):
        super(QuadraticFunctionLayer, self).__init__(**kwargs)
        self.dimension = dimension
        self.W = tf.random.normal([dimension, dimension])
        self.y = tf.random.normal([dimension])

    def build(self, input_shape):
        self.theta = self.add_weight(name="theta", shape=[self.dimension], dtype=tf.float32, initializer="random_normal", trainable=True)
    
    def call(self, inputs=tf.zeros([1])):
        return tf.norm(tf.linalg.matvec(self.W, self.theta) - self.y)
