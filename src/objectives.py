import tensorflow as tf
from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        lrelu = lambda x: keras.activations.relu(x, alpha=0.3)
        self.linear_1 = keras.layers.Dense(32, activation="sigmoid")
        self.linear_2 = keras.layers.Dense(10, activation="sigmoid")

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.linear_2(x)

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
        self.theta = self.add_weight(name="theta", shape=[self.dimension], dtype=tf.float32, initializer=tf.keras.initializers.RandomNormal(), trainable=True)
    
    def call(self, _=tf.zeros([1])):
        return tf.norm(tf.linalg.matvec(self.W, self.theta) - self.y)
