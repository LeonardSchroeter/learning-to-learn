import tensorflow as tf
from tensorflow import keras


# Feedforward neural network with a sigmod activation function in the hidden layer
class MLPSigmoid(keras.layers.Layer):
    def __init__(self):
        super(MLPSigmoid, self).__init__()
        self.linear_1 = keras.layers.Dense(32, activation="sigmoid")
        self.linear_2 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.linear_2(x)

# Feedforward neural network with a ReLU activation function in the hidden layer
class MLPRelu(keras.layers.Layer):
    def __init__(self):
        super(MLPRelu, self).__init__()
        self.linear_1 = keras.layers.Dense(32, activation="relu")
        self.linear_2 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.linear_2(x)

# Feedforward neural network with a leaky ReLU activation function in the hidden layer
class MLPLeakyRelu(keras.layers.Layer):
    def __init__(self):
        super(MLPLeakyRelu, self).__init__()
        lrelu = lambda x: keras.activations.relu(x, alpha=0.1)
        self.linear_1 = keras.layers.Dense(32, activation=lrelu)
        self.linear_2 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.linear_2(x)

# Feedforward neural network with a tanh activation function in the hidden layer
class MLPTanh(keras.layers.Layer):
    def __init__(self):
        super(MLPTanh, self).__init__()
        self.linear_1 = keras.layers.Dense(32, activation="tanh")
        self.linear_2 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.linear_2(x)

# Feedforward neural network with sigmoid activation function in both layers
class MLP(keras.layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = keras.layers.Dense(32, activation="sigmoid")
        self.linear_2 = keras.layers.Dense(10, activation="sigmoid")

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.linear_2(x)

# Convolutional neural network
# This was not used for the tests in the thesis
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

# Quadratic function with parameters as inputs
# This was not used for the tests in the thesis
class QuadraticFunction(keras.layers.Layer):
    def __init__(self, dimension, **kwargs):
        super(QuadraticFunction, self).__init__(**kwargs)
        self.dimension = dimension
        self.W = tf.random.normal([dimension, dimension])
        self.y = tf.random.normal([dimension])

    def call(self, inputs):
        return tf.norm(tf.linalg.matvec(self.W, inputs) - self.y)

# Quadratic function with parameters as weights
# W and y can be set by the user to support testing the same objective function
class QuadraticFunctionLayer(keras.layers.Layer):
    def __init__(self, dimension, same, W, y, **kwargs):
        super(QuadraticFunctionLayer, self).__init__(**kwargs)
        self.dimension = dimension
        if same:
            self.W = W
            self.y = y
        else:
            self.W = tf.random.normal([dimension, dimension])
            self.y = tf.random.normal([dimension])
        self.theta = self.add_weight(name="theta", shape=[self.dimension], dtype=tf.float32, initializer=tf.keras.initializers.RandomNormal(), trainable=True)
    
    def call(self, _=tf.zeros([1])):
        return tf.norm(tf.linalg.matvec(self.W, self.theta) - self.y)
