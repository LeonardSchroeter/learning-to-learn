import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from collections import deque

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

        if "evaluation_size" in config:
            evaluation_size = config["evaluation_size"]
        else: evaluation_size = 0.2

        if "dataset" in config:
            dataset_length = len(list(config["dataset"].as_numpy_iterator()))

            self.dataset_train = config["dataset"].skip(math.floor(dataset_length * evaluation_size))
            self.dataset_test = config["dataset"].take(math.floor(dataset_length * evaluation_size))
        else: raise Exception("Config needs to have a dataset!")

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

        self.util = Util()
        self.objective_network_weights = {}

        self.optimizer_network = self.optimizer_network_generator()

        if self.load_weights:
            self.optimizer_network.load_weights(self.get_checkpoint_path(alternative=self.load_path))

    def setDataset(self, dataset):
        print("Setting dataset")
        dataset_length = len(list(dataset.as_numpy_iterator()))

        self.dataset_train = dataset.skip(math.floor(dataset_length * self.evaluation_size))
        self.dataset_test = dataset.take(math.floor(dataset_length * self.evaluation_size))

    def get_checkpoint_path(self, super_epoch = 0, epoch = 0, alternative = None):
        dirname = os.path.dirname(__file__)
        if alternative:
            filename = alternative
        else:
            filename = f"{super_epoch}_{epoch}"
        relative_path = f"checkpoints\{self.config_name}\{filename}"
        path = os.path.join(dirname, relative_path)
        return path

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

    def new_objective(self, learned_optimizer = False):
        if not learned_optimizer:
            return self.objective_network_generator()

        with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
            objective_network = self.objective_network_generator()
        return objective_network

    def train_optimizer(self):
        print("Train optimizer")

        for super_epoch in range(1, self.super_epochs + 1):
            print("Super epoch: ", super_epoch)

            self.optimizer_network.reset_states()
            self.clear_weights()
            
            objective_network = self.new_objective(learned_optimizer=True)

            steps_left = self.max_steps_per_super_epoch

            for epoch in range(1, self.epochs + 1):
                print("Epoch: ", epoch)

                dataset_train = self.dataset_train.shuffle(buffer_size=1024).batch(self.batch_size)
                dataset_test = self.dataset_test.shuffle(buffer_size=1024).batch(self.batch_size)

                steps_taken = self.train_objective(objective_network, dataset_train, steps_left, True)
                steps_left = steps_left - steps_taken
                
                if epoch % self.evaluate_every_n_epoch == 0:
                    self.evaluate_objective(objective_network, dataset_test)

                if epoch % self.save_every_n_epoch == 0:
                    self.optimizer_network.save_weights(self.get_checkpoint_path(super_epoch, epoch))

                if steps_left == 0:
                    break

        self.optimizer_network.save_weights(self.get_checkpoint_path(alternative="result"))

    def train_objective(self, objective_network, dataset, max_steps_left = math.inf, train_optimizer = False):
        losses = deque(maxlen=self.train_optimizer_steps)

        with tf.GradientTape(persistent=True) as tape:
            for step, (x, y) in dataset.enumerate().as_numpy_iterator():
                if not objective_network.built:
                    with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                        _ = objective_network(x)

                tape.watch(self.objective_network_weights)

                with mock.patch.object(keras.layers.Layer, "__call__", self.custom_call()):
                    outputs = objective_network(x)
                loss = self.objective_loss_fn(y, outputs)
                losses.append(loss)

                with tape.stop_recording():
                    gradients = tape.gradient(loss, self.objective_network_weights)
                    gradients = self.util.to_1d(gradients)
                    gradients = self.objective_gradient_preprocessor(gradients)
                gradients = tf.stop_gradient(gradients)

                optimizer_output = self.optimizer_network(gradients)
                optimizer_output = self.util.from_1d(optimizer_output)
                
                self.apply_weight_changes(optimizer_output)
                
                if step == 0:
                    continue
                elif not train_optimizer:
                    losses.clear()
                    tape.reset()
                elif self.train_optimizer_every_step:
                    optimizer_loss = self.accumulate_losses(losses)
                    with tape.stop_recording():
                        optimizer_gradients = tape.gradient(optimizer_loss, self.optimizer_network.trainable_weights)
                        self.optimizer_optimizer.apply_gradients(zip(optimizer_gradients, self.optimizer_network.trainable_weights))
                elif (step + 1) % self.train_optimizer_steps == 0:
                    optimizer_loss = self.accumulate_losses(losses)
                    with tape.stop_recording():
                        optimizer_gradients = tape.gradient(optimizer_loss, self.optimizer_network.trainable_weights)
                        optimizer_gradients = [tf.math.l2_normalize(grads) for grads in optimizer_gradients]
                        self.optimizer_optimizer.apply_gradients(zip(optimizer_gradients, self.optimizer_network.trainable_weights))
                    losses.clear()
                    tape.reset()

                if step + 1 == max_steps_left:
                    break
        
        return step + 1

    def evaluate_objective(self, objective_network, dataset):
        self.evaluation_metric.reset_state()
        for x, y in dataset.as_numpy_iterator():
            outputs = objective_network(x)
            self.evaluation_metric.update_state(y, outputs)
        print("  Accuracy: ", self.evaluation_metric.result().numpy())
    
    def evaluate_optimizer(self):
        print("Evaluate optimizer")

        self.optimizer_network.reset_states()
        self.clear_weights()
        
        objective_network = self.new_objective(learned_optimizer=True)
        comparison_objectives = [self.new_objective() for _ in self.comparison_optimizers]

        steps_left = self.max_steps_per_super_epoch

        for epoch in range(1, self.epochs + 1):
            print("Epoch: ", epoch)

            dataset_train = self.dataset_train.shuffle(buffer_size=1024).batch(self.batch_size)
            dataset_test = self.dataset_test.shuffle(buffer_size=1024).batch(self.batch_size)

            steps_taken = self.train_objective(objective_network, dataset_train, steps_left)
            for objective, optimizer in zip(comparison_objectives, self.comparison_optimizers):
                self.train_compare(objective, optimizer, dataset_train, steps_left)
            steps_left = steps_left - steps_taken
            
            if epoch % self.evaluate_every_n_epoch == 0:
                self.evaluate_objective(objective_network, dataset_test)
                for objective in comparison_objectives:
                    self.evaluate_objective(objective, dataset_test)
            if steps_left == 0:
                break

    def train_compare(self, objective_network, optimizer, dataset, max_steps_left):
        for step, (x, y) in dataset.enumerate().as_numpy_iterator():
            with tf.GradientTape() as tape:
                outputs = objective_network(x)
                loss = self.objective_loss_fn(y, outputs)

            gradients = tape.gradient(loss, objective_network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, objective_network.trainable_weights))

            if step + 1 == max_steps_left:
                break
    
    def pretrain(self, steps):
        print(f"Pretrain optimizer for {steps} steps")

        # sampling the already preprocessed gradients from a uniform dist from -1 to 1, 
        # since this is the only values the preprocessed gradients can take
        # then take the inverse to derive at the desired outputs, i.e. simulating sgd
        inputs = tf.random.uniform([steps], minval=-1.0, maxval=1.0)
        outputs = preprocess_gradients_inverse(inputs, 10) * -0.001

        # need low stddev, because values need to be similar to inputs in later training
        # inputs = tf.random.normal([steps], mean=0.0, stddev=0.001)
        # outputs = inputs * -1.0
        # inputs = self.objective_gradient_preprocessor(inputs)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        optimizer = keras.optimizers.Adam()

        for x, y in dataset.as_numpy_iterator():
            with tf.GradientTape() as tape:
                out = self.optimizer_network(x)
                loss = keras.losses.mean_squared_error(y, out)
            gradients = tape.gradient(loss, self.optimizer_network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.optimizer_network.trainable_weights))

        objective_network = self.new_objective(learned_optimizer=True)
        x = list(self.dataset_test.batch(self.batch_size).as_numpy_iterator())[0][0]
        with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
            _ = objective_network(x)
        inputs = self.util.to_1d(self.objective_network_weights)
        self.clear_weights()

        weights = self.optimizer_network.get_weights()
        self.optimizer_network = self.optimizer_network_generator()
        _ = self.optimizer_network(inputs)
        self.optimizer_network.set_weights(weights)


def main():
    pass


if __name__ == "__main__":
    main()
