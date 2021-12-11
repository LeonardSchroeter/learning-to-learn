import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from collections import deque

import matplotlib.pyplot as plt
import mock
import tensorflow as tf
from tensorflow import keras

from .util import Util, preprocess_gradients_inverse


class LearningToLearn():
    def __init__(self, config):
        def add_attr(name, default):
            if name in config:
                setattr(self, name, config[name])
            else:
                if default is None:
                    raise Exception(f"Config needs to have property of name: {name}")
                else:
                    setattr(self, name, default)

        add_attr("dataset", None)
        add_attr("batch_size", 64)
        add_attr("optimizer_network_generator", None)
        add_attr("train_optimizer_steps", 16)
        add_attr("objective_network_generator", None)
        add_attr("objective_loss_fn", None)
        add_attr("evaluation_metric", None)
        add_attr("super_epochs", 1)
        add_attr("epochs", 32)
        add_attr("save_every_n_epoch", math.inf)
        add_attr("evaluate_every_n_epoch", 1)
        add_attr("accumulate_losses", tf.add_n)
        add_attr("optimizer_optimizer", keras.optimizers.Adam())
        add_attr("config_name", None)
        add_attr("load_weights", False)
        add_attr("load_path", "result")
        add_attr("comparison_optimizers", [])
        add_attr("max_steps_per_super_epoch", math.inf)
        add_attr("train_optimizer_every_step", False)
        add_attr("objective_gradient_preprocessor", lambda x: x)
        add_attr("evaluation_size", 0.2)
        add_attr("num_layers", 1)
        add_attr("one_optimizer", True)

        self.util = Util()
        self.objective_network_weights = {}

        self.optimizer_networks = []

        if self.one_optimizer:
            num_layers = 1
        else:
            num_layers = self.num_layers
        for _ in range(num_layers):
            self.optimizer_networks.append(self.optimizer_network_generator())

        if self.load_weights:
            for optimizer_network in self.optimizer_networks:
                optimizer_network.load_weights(self.get_checkpoint_path(alternative=self.load_path))

    def get_shuffeled_datasets(self):
        dataset_length = len(self.dataset)
        test_size = math.floor(dataset_length * self.evaluation_size)

        dataset_train = self.dataset.skip(test_size).shuffle(buffer_size=1024).batch(self.batch_size, drop_remainder=True)
        dataset_test = self.dataset.take(test_size).shuffle(buffer_size=1024).batch(self.batch_size, drop_remainder=True)

        return dataset_train, dataset_test

    def get_checkpoint_path(self, super_epoch = 0, epoch = 0, alternative = None):
        if alternative:
            filename = alternative
        else:
            filename = f"{super_epoch}_{epoch}"

        dirname = os.path.dirname(__file__)
        relative_path = f"checkpoints\{self.config_name}\{filename}"
        path = os.path.join(dirname, relative_path)

        return path

    def clear_objective_weights(self):
        self.objective_network_weights = {}

    def reset_optimizer_states(self):
        for optimizer_network in self.optimizer_networks:
            optimizer_network.reset_states()

    def apply_weight_changes(self, g_t):
        for layer_name in self.objective_network_weights.keys():
            for weight_name in self.objective_network_weights[layer_name].keys():
                self.objective_network_weights[layer_name][weight_name] = self.objective_network_weights[layer_name][weight_name] + g_t[layer_name][weight_name]

    def custom_getter_generator(self, layer_name):
        def _custom_getter(weight_name):
            if layer_name in self.objective_network_weights and weight_name in self.objective_network_weights[layer_name]:
                return self.objective_network_weights[layer_name][weight_name]
            return None
        return _custom_getter

    def custom_add_weight(self):
        def _custom_add_weight(other, name, shape, dtype, initializer, **kwargs):
            if initializer:
                tensor = initializer(shape, dtype=dtype)
            else:
                tensor = tf.zeros(shape, dtype=dtype)

            if not other.name in self.objective_network_weights:
                self.objective_network_weights[other.name] = {}

            self.objective_network_weights[other.name][name] = tensor

            custom_getter = self.custom_getter_generator(other.name)
            return custom_getter(name)

        return _custom_add_weight

    def custom_call(self):
        original_call = keras.layers.Layer.__call__
        def _custom_call(other, *args, **kwargs):
            if other.name in self.objective_network_weights:
                for name, weight in self.objective_network_weights[other.name].items():
                    if hasattr(other, name):
                        setattr(other, name, weight)
                    else: print(f"Layer {other.name} does not have attribute {name}")
            return original_call(other, *args, **kwargs)
        return _custom_call

    def new_objective(self, learned_optimizer = False):
        if learned_optimizer:
            with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                objective_network = self.objective_network_generator()
        else:
            objective_network = self.objective_network_generator()
        return objective_network

    def build_objective(self, objective_network, x):
        if not objective_network.built:
            with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                _ = objective_network(x)

    def call_objective(self, objective_network, x):
        with mock.patch.object(keras.layers.Layer, "__call__", self.custom_call()):
            return objective_network(x)

    def train_optimizer(self):
        print("Train optimizer")

        for super_epoch in range(1, self.super_epochs + 1):
            print("Super epoch: ", super_epoch)

            self.reset_optimizer_states()
            self.clear_objective_weights()
            
            objective_network = self.new_objective(learned_optimizer=True)

            steps_left = self.max_steps_per_super_epoch

            for epoch in range(1, self.epochs + 1):
                print("Epoch: ", epoch)

                dataset_train, dataset_test = self.get_shuffeled_datasets()

                steps_left = self.train_objective(objective_network, dataset_train, steps_left, True)

                if epoch % self.evaluate_every_n_epoch == 0:
                    self.evaluate_objective(objective_network, dataset_test)

                if epoch % self.save_every_n_epoch == 0:
                    self.optimizer_networks[0].save_weights(self.get_checkpoint_path(super_epoch, epoch))

                if steps_left == 0:
                    break

        # self.optimizer_networks[0].save_weights(self.get_checkpoint_path(alternative="result"))

    def train_objective(self, objective_network, dataset, steps_left = math.inf, train_optimizer = False, return_losses = False):
        losses = deque(maxlen=self.train_optimizer_steps)
        all_losses = []

        with tf.GradientTape(persistent=True) as tape:
            for step, (x, y) in dataset.enumerate().as_numpy_iterator():
                self.build_objective(objective_network, x)

                tape.watch(self.objective_network_weights)

                outputs = self.call_objective(objective_network, x)

                loss = self.objective_loss_fn(y, outputs)
                with tape.stop_recording():
                    all_losses.append(loss.numpy())
                losses.append(loss)

                if self.one_optimizer:
                    with tape.stop_recording():
                        gradients = tape.gradient(loss, self.objective_network_weights)
                        gradients = self.util.to_1d(gradients)
                        gradients = self.objective_gradient_preprocessor(gradients)
                    gradients = tf.stop_gradient(gradients)

                    optimizer_output = self.optimizer_networks[0](gradients)
                    optimizer_output = self.util.from_1d(optimizer_output)
                else:
                    with tape.stop_recording():
                        gradients = tape.gradient(loss, self.objective_network_weights)
                        gradients = self.util.to_1d_per_layer(gradients)
                        gradients = [self.objective_gradient_preprocessor(g) for g in gradients]
                    gradients = [tf.stop_gradient(g) for g in gradients]

                    optimizer_output = [optimizer_network(g) for optimizer_network, g in zip(self.optimizer_networks, gradients)]
                    optimizer_output = self.util.from_1d_per_layer(optimizer_output)
                self.apply_weight_changes(optimizer_output)

                steps_left -= 1
                if steps_left == 0:
                    break
                
                if not train_optimizer:
                    losses.clear()
                    tape.reset()
                    continue

                if step == 0:
                    continue
                
                if self.train_optimizer_every_step:
                    self.update_optimizer(tape, losses)
                    continue

                if (step + 1) % self.train_optimizer_steps == 0:
                    self.update_optimizer(tape, losses)
                    losses.clear()
                    tape.reset()
                    continue
        
        if return_losses:
            return steps_left, all_losses
        return steps_left

    def update_optimizer(self, tape, losses):
        optimizer_loss = self.accumulate_losses(losses)
        with tape.stop_recording():
            for optimizer_network in self.optimizer_networks:
                optimizer_gradients = tape.gradient(optimizer_loss, optimizer_network.trainable_weights)
                optimizer_gradients = [tf.math.l2_normalize(g) for g in optimizer_gradients]
                self.optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))

    def evaluate_objective(self, objective_network, dataset):
        self.evaluation_metric.reset_state()
        for x, y in dataset.as_numpy_iterator():
            outputs = objective_network(x)
            self.evaluation_metric.update_state(y, outputs)
        print("  Accuracy: ", self.evaluation_metric.result().numpy())
    
    def evaluate_optimizer(self, filename, label = "Loss", clear_figure = True):
        print("Evaluate optimizer")

        self.reset_optimizer_states()
        self.clear_objective_weights()
        
        objective_network = self.new_objective(learned_optimizer=True)
        comparison_objectives = [self.new_objective() for _ in self.comparison_optimizers]

        steps_left = self.max_steps_per_super_epoch

        all_losses = []

        for epoch in range(1, self.epochs + 1):
            print("Epoch: ", epoch)

            dataset_train, dataset_test = self.get_shuffeled_datasets()

            steps_left_new, losses = self.train_objective(objective_network, dataset_train, steps_left, return_losses=True)
            all_losses.extend(losses)

            for objective, optimizer in zip(comparison_objectives, self.comparison_optimizers):
                self.train_compare(objective, optimizer, dataset_train, steps_left)
            
            if epoch % self.evaluate_every_n_epoch == 0:
                self.evaluate_objective(objective_network, dataset_test)
                for objective in comparison_objectives:
                    self.evaluate_objective(objective, dataset_test)

            steps_left = steps_left_new
            if steps_left == 0:
                break

        x = list(range(1, len(all_losses) + 1))

        plt.plot(x, all_losses, label=label)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Loss")

        if filename:
            path = f"plots/{self.config_name}_{filename}.eps"
        else:
            path = f"plots/{self.config_name}.eps"
        plt.savefig(path)

        if clear_figure:
            plt.clf()

    def train_compare(self, objective_network, optimizer, dataset, steps_left):
        for step, (x, y) in dataset.enumerate().as_numpy_iterator():
            if step == steps_left:
                break

            with tf.GradientTape() as tape:
                outputs = objective_network(x)
                loss = self.objective_loss_fn(y, outputs)

            gradients = tape.gradient(loss, objective_network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, objective_network.trainable_weights)) 
    
    def pretrain(self, steps):
        print(f"Pretrain optimizer for {steps} steps")

        # sampling the already preprocessed gradients from a uniform dist from -1 to 1, 
        # since this is the only values the preprocessed gradients can take
        # then take the inverse to derive at the desired outputs, i.e. simulating sgd
        # inputs = tf.random.uniform([steps], minval=-1.0, maxval=1.0)
        # outputs = preprocess_gradients_inverse(inputs, 10) * -0.001

        # need low stddev, because values need to be similar to inputs in later training
        inputs = tf.random.normal([steps], mean=0.0, stddev=0.001)
        outputs = inputs * -1.0
        inputs = self.objective_gradient_preprocessor(inputs)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(self.batch_size, drop_remainder=True)

        optimizer = keras.optimizers.Adam()
        optimizer_network = self.optimizer_network_generator()

        for x, y in dataset.as_numpy_iterator():
            with tf.GradientTape() as tape:
                out = optimizer_network(x)
                loss = keras.losses.mean_squared_error(y, out)
            gradients = tape.gradient(loss, optimizer_network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, optimizer_network.trainable_weights))

        objective_network = self.new_objective(learned_optimizer=True)
        self.train_objective(objective_network, self.get_shuffeled_datasets()[0], 1)

        weights = optimizer_network.get_weights()

        for opt_net in self.optimizer_networks:
            opt_net.set_weights(weights)

        self.clear_objective_weights()
        self.reset_optimizer_states()

def main():
    pass


if __name__ == "__main__":
    main()
