import tensorflow as tf
from tensorflow import keras


class LSTMNetwork(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LSTMNetwork, self).__init__(**kwargs)
        self.lstm = keras.layers.LSTM(10, return_state=True)
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs, hidden_state, cell_state):
        initial_state = [hidden_state, cell_state]
        _, hidden_state, cell_state = self.lstm(inputs, initial_state=initial_state)
        return self.output_layer(hidden_state), hidden_state, cell_state

class LSTMOptimizer(keras.optimizers.Optimizer):
    def __init__(self, lstm_network, name="LSTMOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self.lstm_network = lstm_network
        self.train = True

    def _create_slots(self, var_list):
        for var in var_list:
            number_parameters = tf.size(var).numpy()
            self.add_slot(var, "hidden_state", initializer="zeros", shape=(number_parameters, 10))
            self.add_slot(var, "cell_state", initializer="zeros", shape=(number_parameters, 10))

    def _resource_apply_dense(self, grad, handle):
        number_parameters = tf.size(grad).numpy()
        shape_parameters = tf.shape(grad).numpy()

        hidden_state = self.get_slot(handle, "hidden_state")
        cell_state = self.get_slot(handle, "cell_state")

        parameters = tf.reshape(grad, [number_parameters, 1, 1])

        first = True
        for i, parameter in enumerate(parameters):
            parameter = tf.reshape(parameter, [1, 1, 1])
            output_i, new_hidden_state_i, new_cell_state_i = self.lstm_network(parameter, tf.reshape(hidden_state[i], [1, 10]), tf.reshape(cell_state[i], [1, 10]))

            if(first):
                new_hidden_state = new_hidden_state_i
                new_cell_state = new_cell_state_i
                output = output_i
                first = False
            else:
                new_hidden_state = tf.concat([new_hidden_state, new_hidden_state_i], 0)
                new_cell_state = tf.concat([new_cell_state, new_cell_state_i], 0)
                output = tf.concat([output, output_i], 0)
        
        optimizer_values = tf.reshape(output, shape_parameters)
        new_parameters = handle + optimizer_values
        
        hidden_state.assign(new_hidden_state)
        cell_state.assign(new_cell_state)

        handle.assign(new_parameters)
