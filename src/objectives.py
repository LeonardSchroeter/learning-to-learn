import tensorflow as tf
from tensorflow import keras


class _MLP(keras.layers.Layer):
    def __init__(self, activation_1, activation_2, **kwargs):
        super().__init__(**kwargs)
        self.layer_1 = keras.layers.Dense(32, activation=activation_1)
        self.layer_2 = keras.layers.Dense(10, activation=activation_2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        return self.layer_2(x)

# Feedforward neural network with a sigmod activation function in the hidden layer
class MLPSigmoid(_MLP):
    def __init__(self):
        super().__init__("sigmoid", "softmax")

# Feedforward neural network with a ReLU activation function in the hidden layer
class MLPRelu(_MLP):
    def __init__(self):
        super().__init__("relu", "softmax")

# Feedforward neural network with a leaky ReLU activation function in the hidden layer
class MLPLeakyRelu(_MLP):
    def __init__(self):
        lrelu = lambda x: keras.activations.relu(x, alpha=0.1)
        super().__init__(lrelu, "softmax")

# Feedforward neural network with a tanh activation function in the hidden layer
class MLPTanh(_MLP):
    def __init__(self):
        super().__init__("tanh", "softmax")

# Feedforward neural network with sigmoid activation function in both layers
class MLP(_MLP):
    def __init__(self):
        super().__init__("sigmoid", "sigmoid")

# Convolutional neural network
# This was not used for the tests in the thesis
# class ConvNN(keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
#         self.layer_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.layer_3 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
#         self.layer_4 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.layer_5 = keras.layers.Flatten()
#         self.layer_6 = keras.layers.Dense(10, activation="softmax")

#     def call(self, inputs):
#         x = self.layer_1(inputs)
#         x = self.layer_2(x)
#         x = self.layer_3(x)
#         x = self.layer_4(x)
#         x = self.layer_5(x)
#         return self.layer_6(x)

# Quadratic function with parameters as inputs
# This was not used for the tests in the thesis
# class QuadraticFunction(keras.layers.Layer):
#     def __init__(self, dimension, **kwargs):
#         super().__init__(**kwargs)
#         self.dimension = dimension
#         self.W = tf.random.normal([dimension, dimension])
#         self.y = tf.random.normal([dimension])

#     def call(self, inputs):
#         return tf.norm(tf.linalg.matvec(self.W, inputs) - self.y)

# Quadratic function with parameters as weights
# Initial weights W and y can be set by the user to reproduce the same objective function
class QuadraticFunctionLayer(keras.layers.Layer):
    def __init__(self, dimension, init_weights, W_init, y_init, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        if init_weights:
            self.W = W_init
            self.y = y_init
        else:
            self.W = tf.random.normal([dimension, dimension])
            self.y = tf.random.normal([dimension])
        self.theta = self.add_weight(name="theta", shape=[self.dimension], dtype=tf.float32, initializer=tf.keras.initializers.RandomNormal(), trainable=True)
    
    def call(self, _=tf.zeros([1])):
        return tf.norm(tf.linalg.matvec(self.W, self.theta) - self.y)
