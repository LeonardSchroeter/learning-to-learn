import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import deque

import mock
import tensorflow as tf
from tensorflow import keras

from LSTMNetworkPerParameter import LSTMNetworkPerParameter
from QuadraticFunction import QuadraticFunctionLayer


class LearningToLearn():
    def __init__(self, optimizer_network, objective_network_generator):
        self.optimizer_network = optimizer_network
        self.objective_network_generator = objective_network_generator
        self.objective_network_weights = {}

    def clear_weights(self):
        self.objective_network_weights = {}

    def custom_getter(self, layer_name):
        def _custom_getter(name, **kwargs):
            if layer_name in self.objective_network_weights and name in self.objective_network_weights[layer_name]:
                return self.objective_network_weights[layer_name][name]
            return None
        return _custom_getter

    def custom_add_weight(self):
        original_add_weight = keras.layers.Layer.add_weight
        def _custom_add_weight(other, name, shape, dtype, initializer, **kwargs):
            if initializer:
                tensor = initializer(shape, dtype=dtype)
            else:
                tensor = tf.zeros(shape, dtype=dtype)
            if not other.name in self.objective_network_weights:
                self.objective_network_weights[other.name] = {}
            self.objective_network_weights[other.name][name] = tensor
            getter = self.custom_getter(other.name)
            original_add_weight(other, name=name, shape=shape, dtype=dtype, initializer=initializer, getter=getter, **kwargs)
            return getter(name)
        return _custom_add_weight

    def custom_call(self):
        original_call = keras.layers.Layer.__call__
        def _custom_call(other, *args, **kwargs):
            for name, weight in self.objective_network_weights[other.name].items():
                if hasattr(other, name):
                    setattr(other, name, weight)
            return original_call(other, *args, **kwargs)
        return _custom_call

    def train_optimizer(self, steps = 1_000_000):
        optimizer_optimizer = keras.optimizers.Adam()

        for step in range(steps):
            self.optimizer_network.reset_states()
            self.clear_weights()
            with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                objective_network = self.objective_network_generator()
            optimizer_gradients = self.train_objective(objective_network, 50)
            optimizer_optimizer.apply_gradients(zip(optimizer_gradients, self.optimizer_network.trainable_weights))

    def train_objective(self, objective_network, T = 20):
        losses = deque(maxlen=T)

        with tf.GradientTape(persistent=True) as tape:
            for step in range(T):
                tape.watch(self.objective_network_weights)

                with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                    with mock.patch.object(keras.layers.Layer, "__call__", self.custom_call()):
                        loss = objective_network(tf.zeros([1]))
                
                losses.append(loss)

                with tape.stop_recording():
                    gradients = tape.gradient(loss, self.objective_network_weights)
                # gradients = tf.stop_gradient(gradients)

                gradients = [list(grads.values()) for grads in gradients.values()]
                print(gradients)
                # TODO Fix error where gradients and the optimizer_output have the wrong data structure
                optimizer_output = self.optimizer_network(gradients)

                for (name, weight_t), g_t in (zip(self.objective_network_weights.items(), optimizer_output)):
                    self.objective_network_weights[name] = weight_t + g_t
            optimizer_loss = tf.add_n(losses)
            print(loss.numpy())
        optimizer_gradients = tape.gradient(optimizer_loss, self.optimizer_network.trainable_weights)

        return list(optimizer_gradients)

def main():
    tf.random.set_seed(1)
    objective_network_generator = lambda : QuadraticFunctionLayer(10)
    optimizer_network = LSTMNetworkPerParameter()
    ltl = LearningToLearn(optimizer_network, objective_network_generator)
    ltl.train_optimizer()

if __name__ == "__main__":
    main()
