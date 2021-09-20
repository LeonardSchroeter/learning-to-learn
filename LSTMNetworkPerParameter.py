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

    def size_shape(self, input_dict):
        result = [[(tf.size(tensor).numpy(), tf.shape(tensor).numpy()) for tensor in nested_dict.values()] for nested_dict in input_dict.values()]
        return result
                
    def reconstruct_shape(self, x, sizes_shapes):
        sizes = []

        for arr in sizes_shapes:
            for size, _ in arr:
                sizes.append(size)

        tensors = tf.split(x, sizes, 0)

        result = []
        i = 0
        for arr in sizes_shapes:
            layer_tensors = []
            for _, shape in arr:
                layer_tensors.append(tf.reshape(tensors[i], shape))
                i += 1
            result.append(layer_tensors)

        return result

    def call(self, input_dict):
        sizes_shapes = self.size_shape(input_dict)

        inputs = [list(grads.values()) for grads in input_dict.values()]
        inputs = [tensor for sublist in inputs for tensor in sublist]
        inputs = [tf.reshape(tensor, tf.size(tensor).numpy()) for tensor in inputs]
        inputs = tf.concat(inputs, 0)

        inputs_size = tf.size(inputs).numpy()

        x = self.lstm(tf.reshape(inputs, [inputs_size, 1, 1]))
        x = self.output_layer(x)

        x = tf.reshape(x, [inputs_size])

        x = self.reconstruct_shape(x, sizes_shapes)

        return x
