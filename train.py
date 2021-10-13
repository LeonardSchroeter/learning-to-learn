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


class LearningToLearn():
    def __init__(self, optimizer_network, objective_network_generator, objective_loss_fn, dataset, evaluation_metric, accumulate_losses = tf.add_n):
        self.optimizer_network = optimizer_network
        self.objective_network_generator = objective_network_generator
        self.objective_loss_fn = objective_loss_fn
        self.objective_network_weights = {}
        dataset_length = len(list(dataset.as_numpy_iterator()))
        self.dataset_train = dataset.skip(math.floor(dataset_length * 0.2))
        self.dataset_test = dataset.take(math.floor(dataset_length * 0.2))
        self.evaluation_metric = evaluation_metric
        self.accumulate_losses = accumulate_losses

    def clear_weights(self):
        self.objective_network_weights = {}

    def apply_weight_changes(self, g_t):
        for layer_name in self.objective_network_weights.keys():
            for weight_name in self.objective_network_weights[layer_name].keys():
                self.objective_network_weights[layer_name][weight_name] = self.objective_network_weights[layer_name][weight_name] + g_t[layer_name][weight_name]

    def weights_to_1d_tensor(self, weight_dict):
        sizes_shapes = []
        weights_1d = []
        dict_structure = {}
        for layer_name, layer_weights in weight_dict.items():
            dict_structure[layer_name] = {}
            for weight_name, weights in layer_weights.items():
                dict_structure[layer_name][weight_name] = None
                size = tf.size(weights).numpy()
                shape = tf.shape(weights).numpy()
                sizes_shapes.append((size, shape))
                weights_1d.append(tf.reshape(weights, size))
        all_weights_1d = tf.concat(weights_1d, 0)
        return all_weights_1d, sizes_shapes, dict_structure

    def tensor_1d_to_weights(self, tensor_1d, sizes_shapes, dict_structure):
        result_dict = dict_structure
        sizes, shapes = zip(*sizes_shapes)
        sizes, shapes = list(sizes), list(shapes)
        tensors_split = tf.split(tensor_1d, sizes, 0)
        tensors = [tf.reshape(tensor, shape) for tensor, shape in zip(tensors_split, shapes)]
        i = 0
        for layer_name in result_dict.keys():
            for weight_name in result_dict[layer_name].keys():
                result_dict[layer_name][weight_name] = tensors[i]
                i += 1
        return result_dict

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

    def train_optimizer(self, epochs = 20):
        optimizer_optimizer = keras.optimizers.Adam()

        self.optimizer_network.reset_states()
        self.clear_weights()
        with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
            objective_network = self.objective_network_generator()
        
        for epoch in range(1, epochs + 1):
            print("Epoch: ", epoch)

            dataset_train = self.dataset_train.shuffle(buffer_size=1024).batch(64)
            dataset_test = self.dataset_test.shuffle(buffer_size=1024).batch(64)

            self.train_objective(objective_network, optimizer_optimizer, dataset_train)

            # self.optimizer_network.reset_states()
            # self.clear_weights()
            # with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
            #     objective_network = self.objective_network_generator()
            self.evaluation_metric.reset_state()
            self.evaluate_optimizer(objective_network, dataset_test)
            print("  Accuracy: ", self.evaluation_metric.result().numpy(), end="\n\n")

    def train_objective(self, objective_network, optimizer_optimizer, dataset, T = 16):
        losses = deque(maxlen=T)

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
                gradients, sizes_shapes, dict_structure = self.weights_to_1d_tensor(gradients)
                gradients = tf.stop_gradient(gradients)

                optimizer_output = self.optimizer_network(gradients)
                optimizer_output = self.tensor_1d_to_weights(optimizer_output, sizes_shapes, dict_structure)
                
                self.apply_weight_changes(optimizer_output)
                
                if (step + 1) % T == 0:
                    optimizer_loss = self.accumulate_losses(losses)
                    with tape.stop_recording():
                        optimizer_gradients = tape.gradient(optimizer_loss, self.optimizer_network.trainable_weights)
                        optimizer_optimizer.apply_gradients(zip(optimizer_gradients, self.optimizer_network.trainable_weights))
                    losses.clear()
                    tape.reset()

    def evaluate_optimizer(self, objective_network, dataset):
        for step, (x, y) in dataset.enumerate().as_numpy_iterator():
            if not objective_network.built:
                with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                    building_output = objective_network(x)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.objective_network_weights)

                with mock.patch.object(keras.layers.Layer, "__call__", self.custom_call()):
                    outputs = objective_network(x)

                loss = self.objective_loss_fn(y, outputs)

            self.evaluation_metric.update_state(y, outputs)

            gradients = tape.gradient(loss, self.objective_network_weights)
            gradients, sizes_shapes, dict_structure = self.weights_to_1d_tensor(gradients)

            optimizer_output = self.optimizer_network(gradients)
            optimizer_output = self.tensor_1d_to_weights(optimizer_output, sizes_shapes, dict_structure)

            self.apply_weight_changes(optimizer_output)


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
    objective_network_generator = lambda : ConvNN()
    objective_loss_fn = keras.losses.SparseCategoricalCrossentropy()
    evaluation_metric = keras.metrics.SparseCategoricalAccuracy()

    optimizer_network = LSTMNetworkPerParameter()
    ltl = LearningToLearn(optimizer_network, objective_network_generator, objective_loss_fn, dataset, evaluation_metric)
    ltl.train_optimizer(100)

if __name__ == "__main__":
    main()
