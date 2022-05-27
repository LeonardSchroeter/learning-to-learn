import copy
import math
from collections import deque

import mock
import tensorflow as tf
from tensorflow import keras

from .util import Util


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

        # initialize attributes with default values if not specified in config
        add_attr("objective_network_generator", None)
        add_attr("num_layers", 1)
        add_attr("objective_loss_fn", None)
        add_attr("objective_gradient_preprocessor", lambda x: x)

        add_attr("dataset", None)
        add_attr("evaluation_size", 0.2)
        add_attr("batch_size", 64)

        add_attr("optimizer_network_generator", None)
        add_attr("one_optimizer", True)
        add_attr("preprocess_optimizer_gradients", True)
        add_attr("optimizer_optimizer", keras.optimizers.Adam())
        add_attr("train_optimizer_steps", 16)
        add_attr("accumulate_losses", tf.add_n)
        add_attr("train_optimizer_every_step", False)

        add_attr("super_epochs", 1)
        add_attr("epochs", 32)
        add_attr("max_steps_per_super_epoch", math.inf)

        add_attr("evaluate_every_n_epoch", 1)
        add_attr("evaluation_metric", None)

        add_attr("comparison_optimizers", [])

        self.util = Util()
        self.objective_network_weights = {}

        self.optimizer_networks = []

        if self.one_optimizer:
            num_layers = 1
        else:
            num_layers = self.num_layers
        for _ in range(num_layers):
            self.optimizer_networks.append(self.optimizer_network_generator())

    # shuffle the dataset
    def get_shuffeled_datasets(self):
        dataset_length = len(self.dataset)
        test_size = math.floor(dataset_length * self.evaluation_size)

        dataset_train = self.dataset.skip(test_size).shuffle(buffer_size=1024).batch(self.batch_size, drop_remainder=True)
        dataset_test = self.dataset.take(test_size).shuffle(buffer_size=1024).batch(self.batch_size, drop_remainder=True)

        return dataset_train, dataset_test

    # clear the objective parameter dictionary
    # should be called before building a new objective network
    def clear_objective_weights(self):
        self.objective_network_weights = {}

    # reset internal states (hidden and cell state) of the optimizer networks
    def reset_optimizer_states(self):
        for optimizer_network in self.optimizer_networks:
            optimizer_network.reset_states()

    # add steps to the objective parameter dictionary
    def apply_weight_changes(self, g_t):
        for layer_name in self.objective_network_weights.keys():
            for weight_name in self.objective_network_weights[layer_name].keys():
                self.objective_network_weights[layer_name][weight_name] = self.objective_network_weights[layer_name][weight_name] + g_t[layer_name][weight_name]

    # returns a custom getter to return custom weights
    def custom_getter_generator(self, layer_name):
        def _custom_getter(weight_name):
            if layer_name in self.objective_network_weights and weight_name in self.objective_network_weights[layer_name]:
                return self.objective_network_weights[layer_name][weight_name]
            return None
        return _custom_getter

    # returns a custom add_weight function that adds a weight tensor inside the objective parameter dictionary every time weights are added
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

    # returns a custom call function that overwrites the objective weights with the weights stored in the objective parameter dictionary before calling the original call function
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

    # generate a new objective function, also adds the weights to the objective parameter dictionary using the custom add_weight function
    def new_objective(self, learned_optimizer = False, same = False):
        if learned_optimizer:
            with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                objective_network = self.objective_network_generator(same)
        else:
            objective_network = self.objective_network_generator(same)
        return objective_network

    # builds the current objective to initialize all missing weights and add them to the objective parameter dictionary
    def build_objective(self, objective_network, x):
        if not objective_network.built:
            with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                _ = objective_network(x)

    # calls the objective function with the custom call function
    def call_objective(self, objective_network, x):
        with mock.patch.object(keras.layers.Layer, "__call__", self.custom_call()):
            return objective_network(x)

    # main training function that loops over multiple super epochs and epochs
    def train_optimizer(self):
        print("Train optimizer")

        for super_epoch in range(1, self.super_epochs + 1):
            print("Super epoch: ", super_epoch)

            self.reset_optimizer_states()
            self.clear_objective_weights()
            
            objective_network = self.new_objective(learned_optimizer=True)

            # store the number of steps left for the current epoch
            steps_left = self.max_steps_per_super_epoch

            for epoch in range(1, self.epochs + 1):
                print("Epoch: ", epoch)

                dataset_train, dataset_test = self.get_shuffeled_datasets()

                steps_left = self.train_objective(objective_network, dataset_train, steps_left, True)

                if epoch % self.evaluate_every_n_epoch == 0:
                    self.evaluate_objective(objective_network, dataset_test)

                if steps_left == 0:
                    break

    # train the objective function for a given number of steps or the length of the dataset
    def train_objective(self, objective_network, dataset, steps_left = math.inf, train_optimizer = False, return_losses = False):
        losses = deque(maxlen=self.train_optimizer_steps)
        all_losses = []

        with tf.GradientTape(persistent=True) as tape:
            for step, (x, y) in dataset.enumerate().as_numpy_iterator():
                if steps_left == 0:
                    break

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
                        # transform the gradients to a flat tensor to be able to pass them to the gradient preprocessor
                        gradients = self.util.to_1d(gradients)
                        gradients = self.objective_gradient_preprocessor(gradients)
                    gradients = tf.stop_gradient(gradients)

                    optimizer_output = self.optimizer_networks[0](gradients)
                    optimizer_output = self.util.from_1d(optimizer_output)
                else:
                    with tape.stop_recording():
                        gradients = tape.gradient(loss, self.objective_network_weights)
                        # transform the gradients to a flat tensor to be able to pass them to the gradient preprocessor
                        gradients = self.util.to_1d_per_layer(gradients)
                        gradients = [self.objective_gradient_preprocessor(g) for g in gradients]
                    gradients = [tf.stop_gradient(g) for g in gradients]

                    optimizer_output = [optimizer_network(g) for optimizer_network, g in zip(self.optimizer_networks, gradients)]
                    optimizer_output = self.util.from_1d_per_layer(optimizer_output)
                self.apply_weight_changes(optimizer_output)

                steps_left -= 1
                
                if not train_optimizer:
                    losses.clear()
                    tape.reset()
                    continue
                
                # cannot train the optimizer in the first step as the loss does not yet depend on the optimizer parameters
                if step == 0:
                    continue
                
                # update optimizer every step
                if self.train_optimizer_every_step:
                    self.update_optimizer(tape, losses)
                    continue
                
                # update optimizer every T steps
                if (step + 1) % self.train_optimizer_steps == 0:
                    self.update_optimizer(tape, losses)
                    losses.clear()
                    tape.reset()
                    continue
        
        if return_losses:
            # return losses for plotting later
            return steps_left, all_losses
        return steps_left

    # update optimizer parameters
    def update_optimizer(self, tape, losses):
        # build loss function for the optimizer
        optimizer_loss = self.accumulate_losses(losses)
        with tape.stop_recording():
            for optimizer_network in self.optimizer_networks:
                optimizer_gradients = tape.gradient(optimizer_loss, optimizer_network.trainable_weights)
                if self.preprocess_optimizer_gradients:
                    # use euclidean norm to normalize the gradients of the optimizer parameters
                    optimizer_gradients = [tf.math.l2_normalize(g) for g in optimizer_gradients]
                self.optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))

    # evaluate objective function
    def evaluate_objective(self, objective_network, dataset):
        self.evaluation_metric.reset_state()
        for x, y in dataset.as_numpy_iterator():
            outputs = objective_network(x)
            self.evaluation_metric.update_state(y, outputs)
        print("  Accuracy: ", self.evaluation_metric.result().numpy())
    
    # evaluate optimizer by training the objective function with the optimizer and evaluating the objective function
    def evaluate_optimizer(self, objective_network_weights = None):
        print("Evaluate optimizer")

        self.reset_optimizer_states()
        self.clear_objective_weights()
        
        objective_network = self.new_objective(learned_optimizer=True, same=True)

        # set the initial weights of the objective function if provided to make comparison more meaningful
        if objective_network_weights:
            objective_network_weights = list(objective_network_weights.values())
            for layer in self.objective_network_weights.keys():
                self.objective_network_weights[layer] = objective_network_weights[0]
                objective_network_weights = objective_network_weights[1:]
        
        # copy initial weights to later return them to use them as initial weights for other learned optimizers
        weights_copy = copy.deepcopy(self.objective_network_weights)

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

        return all_losses, weights_copy

    # train objective network with different optimizers for comparison
    # these results are not used in the thesis
    def train_compare(self, objective_network, optimizer, dataset, steps_left):
        for step, (x, y) in dataset.enumerate().as_numpy_iterator():
            if step == steps_left:
                break

            with tf.GradientTape() as tape:
                outputs = objective_network(x)
                loss = self.objective_loss_fn(y, outputs)

            gradients = tape.gradient(loss, objective_network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, objective_network.trainable_weights)) 
    
    # pretrain the optimizer
    def pretrain(self, steps):
        print(f"Pretrain optimizer for {steps} steps")

        self.reset_optimizer_states()
        self.clear_objective_weights()

        # need low stddev because values need to be similar to inputs in later training
        inp = tf.random.normal([steps], mean=0.0, stddev=0.001)
        # hardcoded learning rate, this is fine because we only test pretraining with the same learning rate
        outputs = inp * -0.01
        inputs = self.objective_gradient_preprocessor(inp)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(self.batch_size, drop_remainder=True)

        optimizer = keras.optimizers.Adam()
        optimizer_network = self.optimizer_network_generator()

        # loss function that punished different signs much more than different magnitudes
        # also explained in the thesis
        loss_fn = lambda y_true, y_pred: tf.math.reduce_mean(tf.math.abs(y_true - y_pred) * tf.exp(-2 * tf.sign(y_true * y_pred)))

        for x, y in dataset.as_numpy_iterator():
            with tf.GradientTape() as tape:
                out = optimizer_network(x)
                loss = loss_fn(y, out)
            gradients = tape.gradient(loss, optimizer_network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, optimizer_network.trainable_weights))

        objective_network = self.new_objective(learned_optimizer=True)
        self.train_objective(objective_network, self.get_shuffeled_datasets()[0], 1)

        weights = optimizer_network.get_weights()

        # set weights of all optimizers to the weights of the pretrained optimizer
        for opt_net in self.optimizer_networks:
            opt_net.set_weights(weights)

        self.clear_objective_weights()
        self.reset_optimizer_states()

def main():
    pass


if __name__ == "__main__":
    main()
