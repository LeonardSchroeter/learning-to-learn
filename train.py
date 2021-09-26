import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import deque

import mock
import tensorflow as tf
from tensorflow import keras

from examples import ConvNN
from optimizer_network import LSTMNetworkPerParameter
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

        for epoch in range(1, epochs + 1):
            print("Epoch: ", epoch)
            dataset = self.dataset.shuffle(buffer_size=1024).batch(64)
            self.optimizer_network.reset_states()
            self.clear_weights()
            with mock.patch.object(keras.layers.Layer, "add_weight", self.custom_add_weight()):
                objective_network = self.objective_network_generator()
            self.train_objective(objective_network, optimizer_optimizer, dataset)

    def train_objective(self, objective_network, optimizer_optimizer, dataset, T = 16):
        losses = deque(maxlen=T)

        metric = keras.metrics.SparseCategoricalAccuracy()

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

                # TODO All outputs are zero after a small amount of steps, but the bias of the dense layer.
                # Find out why this happends, this is likely the cause of the network not seeming to learn.
                # There seems to be a dependency though, since they are 0 and not None
                with tape.stop_recording():
                    gradients = tape.gradient(loss, self.objective_network_weights)

                gradients, sizes_shapes, dict_structure = self.weights_to_1d_tensor(gradients)

                # TODO Find out whether this line is needed.
                # I think it doesn't, since computing the gradients inside stop_recording
                # is enough to make the tape not track the gradients.
                # Simple experiments of running the code with and without this line
                # give the same results supporting this assumption.
                gradients = tf.stop_gradient(gradients)

                optimizer_output = self.optimizer_network(gradients)

                optimizer_output = self.tensor_1d_to_weights(optimizer_output, sizes_shapes, dict_structure)

                self.apply_weight_changes(optimizer_output)
                
                if (step + 1) % T == 0:
                    metric.reset_state()
                    metric.update_state(y, outputs)
                    print("  Loss: ", loss.numpy())
                    print("  Accuracy: ", metric.result().numpy())
                    print("______________________________")
                    optimizer_loss = self.accumulate_losses(losses)
                    with tape.stop_recording():
                        optimizer_gradients = tape.gradient(optimizer_loss, self.optimizer_network.trainable_weights)
                        # TODO Optimizer gradients are too small for ADAM to significantly change the optimizers weights
                        # new_grads = []
                        # for i in range(len(optimizer_gradients) - 2):
                        #     new_grads.append(tf.math.scalar_mul(1e20, optimizer_gradients[i]))
                        # new_grads.append(optimizer_gradients[-2])
                        # new_grads.append(optimizer_gradients[-1])
                        optimizer_optimizer.apply_gradients(zip(optimizer_gradients, self.optimizer_network.trainable_weights))
                    losses.clear()
                    tape.reset()

def main():
    tf.random.set_seed(1)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 28, 28, 1).astype("float32") / 255, y_train)
    )
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (tf.zeros([60000, 784]), tf.zeros([60000]))
    # )

    # objective_network_generator = lambda : QuadraticFunctionLayer(10)
    objective_network_generator = lambda : ConvNN()
    # objective_loss_fn = lambda x, y: y
    objective_loss_fn = keras.losses.SparseCategoricalCrossentropy()
    optimizer_network = LSTMNetworkPerParameter()
    ltl = LearningToLearn(optimizer_network, objective_network_generator, objective_loss_fn, dataset)
    ltl.train_optimizer()

if __name__ == "__main__":
    main()
