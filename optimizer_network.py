import tensorflow as tf
from tensorflow import keras


class LSTMNetworkPerParameter(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LSTMNetworkPerParameter, self).__init__(**kwargs)
        self.lstm = keras.layers.LSTM(10, stateful=True)
        self.output_layer = keras.layers.Dense(1)

    def reset_states(self):
        if self.lstm.built:
            self.lstm.reset_states()

    def call(self, inputs):
        size = tf.size(inputs).numpy()

        x = self.lstm(tf.reshape(inputs, [size, 1, 1]))
        x = self.output_layer(x)
        return tf.reshape(x, [size])
