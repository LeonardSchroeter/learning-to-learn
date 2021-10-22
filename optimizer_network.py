import tensorflow as tf
from tensorflow import keras


class LSTMNetworkPerParameter(keras.Model):
    def __init__(self, learning_rate = 1, **kwargs):
        super(LSTMNetworkPerParameter, self).__init__(**kwargs)
        self.lstm1 = keras.layers.LSTM(20, stateful=True, return_sequences=True)
        self.lstm2 = keras.layers.LSTM(20, stateful=True)
        # layer to combine outputs to a scalar, trainable
        self.output_layer = keras.layers.Dense(1, use_bias=False)
        # layer to combine outputs to a scalar by summing them, not trainable
        # for some reason this configuration learns the opposite of what it should learn, the accuracy goes towards zero
        # for some reason this does not yet seem to work
        # self.output_layer = keras.layers.Dense(1, use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.Constant(value=0.01))

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
