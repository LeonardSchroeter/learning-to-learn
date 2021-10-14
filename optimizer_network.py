import tensorflow as tf
from tensorflow import keras


class LSTMNetworkPerParameter(keras.Model):
    def __init__(self, **kwargs):
        super(LSTMNetworkPerParameter, self).__init__(**kwargs)
        self.lstm1 = keras.layers.LSTM(20, stateful=True, return_sequences=True)
        self.lstm2 = keras.layers.LSTM(20, stateful=True)
        self.output_layer = keras.layers.Dense(1)

    def reset_states(self):
        if self.lstm1.built and self.lstm2.built:
            self.lstm1.reset_states()
            self.lstm2.reset_states()

    def call(self, inputs):
        size = tf.size(inputs).numpy()

        x = tf.reshape(inputs, [size, 1, 1])
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = 0.01 * self.output_layer(x)
        return tf.reshape(x, [size])
