import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from collections import deque

import mock
import tensorflow as tf
from tensorflow import keras

from examples import ConvNN
from optimizer_network import LSTMNetworkPerParameter
from QuadraticFunction import QuadraticFunctionLayer
from util import Util


class LearningToLearn():
    def __init__(self, config):
        def add_attr(name, default):
            if name in config:
                setattr(self, name, config[name])
            else:
                if default is None:
                    raise Exception(f"Config need to have property of name: {name}")
                else:
                    setattr(self, name, default)

        if "evaluation_size" in config:
            evaluation_size = config["evaluation_size"]
        else: evaluation_size = 0.2

        if "dataset" in config:
            dataset_length = len(list(config["dataset"].as_numpy_iterator()))

            self.dataset_train = config["dataset"].skip(math.floor(dataset_length * evaluation_size))
            self.dataset_test = config["dataset"].take(math.floor(dataset_length * evaluation_size))
        else: raise Exception("Config needs to have a dataset!")

        add_attr("batch_size", 64)
        add_attr("optimizer_network", None)
        add_attr("train_optimizer_steps", 16)
        add_attr("objective_network_generator", None)
        add_attr("objective_loss_fn", None)
        add_attr("evaluation_metric", None)
        add_attr("super_epochs", 1)
        add_attr("epochs", 32)
        add_attr("save_every_n_epoch", 0)
        add_attr("evaluate_every_n_epoch", 0)
        add_attr("accumulate_losses", tf.add_n)
        add_attr("optimizer_optimizer", keras.optimizers.Adam())

        self.util = Util()
        self.objective_network_weights = {}

    def clear_weights(self):
        self.objective_network_weights = {}

    def apply_weight_changes(self, g_t):
        for layer_name in self.objective_network_weights.keys():
            for weight_name in self.objective_network_weights[layer_name].keys():
                self.objective_network_weights[layer_name][weight_name] = self.objective_network_weights[layer_name][weight_name] + g_t[layer_name][weight_name]

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

    def new_objective(self):
        with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
            objective_network = self.objective_network_generator()
        return objective_network

    def train_optimizer(self):
        for super_epoch in range(1, self.super_epochs + 1):
            print("Starting training of fresh objective: ", super_epoch)

            self.optimizer_network.reset_states()
            self.clear_weights()
            
            objective_network = self.new_objective()

            for epoch in range(1, self.epochs + 1):
                print("Epoch: ", epoch)

                dataset_train = self.dataset_train.shuffle(buffer_size=1024).batch(self.batch_size)
                dataset_test = self.dataset_test.shuffle(buffer_size=1024).batch(self.batch_size)

                self.train_objective(objective_network, dataset_train)
                
                if epoch % self.evaluate_every_n_epoch == 0:
                    self.evaluation_metric.reset_state()
                    self.evaluate_objective(objective_network, dataset_test)
                    print("  Accuracy: ", self.evaluation_metric.result().numpy())

    def train_objective(self, objective_network, dataset):
        losses = deque(maxlen=self.train_optimizer_steps)

        with tf.GradientTape(persistent=True) as tape:
            for step, (x, y) in dataset.enumerate().as_numpy_iterator():
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
                gradients = self.util.to_1d(gradients)
                gradients = tf.stop_gradient(gradients)

                optimizer_output = self.optimizer_network(gradients)
                optimizer_output = self.util.from_1d(optimizer_output)
                
                self.apply_weight_changes(optimizer_output)
                
                if (step + 1) % self.train_optimizer_steps == 0:
                    optimizer_loss = self.accumulate_losses(losses)
                    with tape.stop_recording():
                        optimizer_gradients = tape.gradient(optimizer_loss, self.optimizer_network.trainable_weights)
                        self.optimizer_optimizer.apply_gradients(zip(optimizer_gradients, self.optimizer_network.trainable_weights))
                    losses.clear()
                    tape.reset()

    def evaluate_objective(self, objective_network, dataset):
        for step, (x, y) in dataset.enumerate().as_numpy_iterator():
            if not objective_network.built:
                with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                    building_output = objective_network(x)
            
            outputs = objective_network(x)

            self.evaluation_metric.update_state(y, outputs)

class QuadMetric():
    def __init__(self):
        self.last_loss = tf.zeros([1])

    def reset_state(self):
        self.last_loss = tf.zeros([1])

    def update_state(self, inputs, outputs):
        self.last_loss = outputs

    def result(self):
        return self.last_loss

def main():
    tf.random.set_seed(1)

    # Quadratic example
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (tf.zeros([2000]), tf.zeros([2000]))
    # )
    # objective_network_generator = lambda : QuadraticFunctionLayer(10)
    # objective_loss_fn = lambda x, y: y
    # evaluation_metric = QuadMetric()

    # MNIST example
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 28, 28, 1).astype("float32") / 255, y_train)
    )

    # There happens an error after 20 epochs with this config
    # config = {
    #     "dataset": dataset,
    #     "batch_size": 64,
    #     "evaluation_size": 0.2,
    #     "optimizer_network": LSTMNetworkPerParameter(),
    #     "optimizer_optimizer": keras.optimizers.Adam(),
    #     "train_optimizer_steps": 16,
    #     "objective_network_generator": lambda : ConvNN(),
    #     "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
    #     "accumulate_losses": tf.add_n,
    #     "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),
    #     "super_epochs": 1,
    #     "epochs": 100,
    #     "save_every_n_epoch": 5,
    #     "evaluate_every_n_epoch": 1,
    # }

    config = {
        "dataset": dataset,
        "batch_size": 64,
        "evaluation_size": 0.2,
        "optimizer_network": LSTMNetworkPerParameter(),
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "objective_network_generator": lambda : ConvNN(),
        "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
        "accumulate_losses": tf.add_n,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),
        "super_epochs": 10,
        "epochs": 10,
        "save_every_n_epoch": 5,
        "evaluate_every_n_epoch": 1,
    }

    ltl = LearningToLearn(config)
    ltl.train_optimizer()


if __name__ == "__main__":
    main()
