import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import deque

import mock
import tensorflow as tf
from tensorflow import keras

from examples import MLP
from LSTMNetworkPerParameter import LSTMNetworkPerParameter
from QuadraticFunction import QuadraticFunctionLayer


class LearningToLearn():
    def __init__(self, optimizer_network, objective_network_generator, objective_loss_fn, dataset, accumulate_losses = tf.add_n):
        self.optimizer_network = optimizer_network
        self.objective_network_generator = objective_network_generator
        self.objective_loss_fn = objective_loss_fn
        self.objective_network_weights = {}
        self.dataset = dataset
        self.accumulate_losses = accumulate_losses

    def clear_weights(self):
        self.objective_network_weights = {}

    def apply_weight_changes(self, g_t):
        for layer, g_t_layer in zip(self.objective_network_weights.keys(), g_t):
            for name, g_t_tensor in zip(self.objective_network_weights[layer].keys(), g_t_layer):
                self.objective_network_weights[layer][name] = self.objective_network_weights[layer][name] + 0.01 * g_t_tensor

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
            if other.name in self.objective_network_weights:
                for name, weight in self.objective_network_weights[other.name].items():
                    if hasattr(other, name):
                        setattr(other, name, weight)
            return original_call(other, *args, **kwargs)
        return _custom_call

    def train_optimizer(self):
        optimizer_optimizer = keras.optimizers.Adam()

        for step, batch in self.dataset.enumerate():
            print("Step: ", step.numpy())
            self.optimizer_network.reset_states()
            self.clear_weights()
            with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                objective_network = self.objective_network_generator()
            self.train_objective(objective_network, optimizer_optimizer, batch)

    def train_objective(self, objective_network, optimizer_optimizer, batch, T = 16):
        losses = deque(maxlen=T)

        with tf.GradientTape(persistent=True) as tape:
            for step, (x, y) in tf.data.Dataset.from_tensor_slices(batch).enumerate().as_numpy_iterator():
                if not objective_network.built:
                    with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                        building_output = objective_network(x)

                tape.watch(self.objective_network_weights)

                with mock.patch.object(keras.layers.Layer, "__call__", self.custom_call()):
                    outputs = objective_network(x)

                loss = self.objective_loss_fn(y, outputs)

                losses.append(loss)

                with tape.stop_recording():
                    gradients = tape.gradient(loss, self.objective_network_weights)

                # TODO Find out whether this line is needed.
                # I think it doesn't, since computing the gradients inside stop_recording
                # is enough to make the tape not track the gradients.
                # Simple experiments of running the code with and without this line
                # give the same results supporting this assumption.
                # gradients = tf.stop_gradient(gradients)

                optimizer_output = self.optimizer_network(gradients)
                self.apply_weight_changes(optimizer_output)
                
                if (step + 1) % T == 0:
                    print("  Loss: ", loss.numpy())
                    optimizer_loss = self.accumulate_losses(losses)
                    with tape.stop_recording():
                        optimizer_gradients = tape.gradient(optimizer_loss, self.optimizer_network.trainable_weights)
                        optimizer_optimizer.apply_gradients(zip(optimizer_gradients, self.optimizer_network.trainable_weights))
                    losses.clear()
                    tape.reset()


def main():
    tf.random.set_seed(1)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
    )
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (tf.zeros([60000, 784]), tf.zeros([60000]))
    # )
    dataset = dataset.shuffle(buffer_size=1024).batch(1).batch(2048)

    # objective_network_generator = lambda : QuadraticFunctionLayer(10)
    objective_network_generator = lambda : MLP()
    # objective_loss_fn = lambda x, y: y
    objective_loss_fn = keras.losses.SparseCategoricalCrossentropy()
    optimizer_network = LSTMNetworkPerParameter()
    ltl = LearningToLearn(optimizer_network, objective_network_generator, objective_loss_fn, dataset)
    ltl.train_optimizer()

if __name__ == "__main__":
    main()
