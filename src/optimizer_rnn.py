import tensorflow as tf
from tensorflow import keras


# Optimizer network structure
# dense_trainable = False was used for all tests in the thesis
class LSTMNetworkPerParameter(keras.Model):
    def __init__(self, learning_rate = 1, dense_trainable = True, **kwargs):
        super(LSTMNetworkPerParameter, self).__init__(**kwargs)
        self.lstm1 = keras.layers.LSTM(20, stateful=True, return_sequences=True)
        self.lstm2 = keras.layers.LSTM(20, stateful=True)
        if dense_trainable:
            # layer to combine outputs to a scalar, trainable
            self.output_layer = keras.layers.Dense(1, use_bias=False)
        else:
            # layer to combine outputs to a scalar by summing them, not trainable
            self.output_layer = keras.layers.Dense(1, use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.Constant(value=1.0))

        self.learning_rate = learning_rate

    def reset_states(self):
        if self.lstm1.built and self.lstm2.built:
            self.lstm1.reset_states()
            self.lstm2.reset_states()

    def call(self, inputs):
        size = tf.size(inputs).numpy()

        x = tf.reshape(inputs, [size, 1, 1])
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.learning_rate * self.output_layer(x)
        return tf.reshape(x, [size])
