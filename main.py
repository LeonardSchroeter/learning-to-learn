import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.custom_metrics import QuadMetric
from src.objectives import MLP, ConvNN, QuadraticFunctionLayer
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.train import LearningToLearn
from src.util import preprocess_gradients

################################################################################
# config parameters
################################################################################
# config_name - name of the config - required
# objective_network_generator - callable that returns a network for training the objective - required
# num_layers - number of layers in the objective network - default: 1
# objective_loss_fn - a tensorflow loss function - required
# objective_gradient_preprocessor - callable that preprocesses the objective gradient - default: lambda x: x

# dataset - dataset - required
# evaluation_size - fraction of dataset to use for evaluation - default 0.2
# batch_size - size of batches for training - default: 64

# optimizer_network_generator - callable that returns a network for training the optimizer - required
# one_optimizer - boolean - if true, use one optimizer for all layers - default: True
# optimizer_optimizer - a tensorflow optimizer for optimizing the optimizer - default: keras.optimizers.Adam()
# train_optimizer_steps - number of steps the loss is accumlated to train the optimizer - default: 16
# accumulate_losses - a tensorflow function that accumulates the losses for training the optimizer - default: tf.add_n
# train_optimizer_every_step - boolean whether to train the optimizer every step - default: False

# super_epochs - number of super epochs - default: 1
# epochs - number of epochs per super epoch - default: 32
# max_steps_per_super_epoch - maximum number of steps per super epoch - default: math.inf

# evaluate_every_n_epoch - evaluate the model every n epochs - default: 1
# evaluation_metric - a tensorflow metric object - required

# save_every_n_epoch - save the model every n epochs - default: math.inf
# load_weights - boolean whether to load weights from a previous run - default: False
# load_path - name of file to load weights from - default: "result"

# comparison_optimizers - list of optimizers to compare with - default: []
################################################################################

# empty config which can be used to create a new config
config_empty = {
    "config_name": None,
    "objective_network_generator": None,
    "num_layers": 1,
    "objective_loss_fn": None,
    "objective_gradient_preprocessor": lambda x: x,

    "dataset": None,
    "evaluation_size": 0.2,
    "batch_size": 64,

    "optimizer_network_generator": None,
    "one_optimizer": True,
    "optimizer_optimizer": keras.optimizers.Adam(),
    "train_optimizer_steps": 16,
    "accumulate_losses": tf.add_n,
    "train_optimizer_every_step": False,

    "super_epochs": 1,
    "epochs": 32,
    "max_steps_per_super_epoch": math.inf,

    "evaluate_every_n_epoch": 1,
    "evaluation_metric": None,

    "save_every_n_epoch": math.inf,
    "load_weights": False,
    "load_path": "result",

    "comparison_optimizers": [],
}

(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
mnist_dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 28, 28, 1).astype("float32") / 255, y_train))

(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
mnist_dnn_dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255, y_train))

(x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
fashion_mnist_dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 28, 28, 1).astype("float32") / 255, y_train))

quadratic_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([640]), tf.zeros([640])))

# config for mnist no preprocessing
# good seed: 8
config_mnist = {
    "config_name": "mnist",
    "objective_network_generator": lambda: ConvNN(),
    "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
    "objective_gradient_preprocessor": lambda x: x,

    "dataset": mnist_dataset,
    "evaluation_size": 0.2,
    "batch_size": 64,

    "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.1),
    "optimizer_optimizer": keras.optimizers.Adam(),
    "train_optimizer_steps": 16,
    "accumulate_losses": tf.add_n,
    "train_optimizer_every_step": False,

    "super_epochs": 1,
    "epochs": 1,
    "max_steps_per_super_epoch": math.inf,

    "evaluate_every_n_epoch": 1,
    "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

    "save_every_n_epoch": math.inf,
    "load_weights": False,
    "load_path": "result",

    "comparison_optimizers": [keras.optimizers.Adam()],
}

config_mnist_2 = {
    "config_name": "mnist_relu_sigmoid",
    "objective_network_generator": lambda: ConvNN(),
    "num_layers": 3,
    "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
    "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),

    "dataset": mnist_dataset,
    "evaluation_size": 0.2,
    "batch_size": 128,

    "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01, dense_trainable=False),
    "one_optimizer": False,
    "optimizer_optimizer": keras.optimizers.Adam(),
    "train_optimizer_steps": 16,
    "accumulate_losses": tf.add_n,
    "train_optimizer_every_step": False,

    "super_epochs": 5,
    "epochs": 1,
    "max_steps_per_super_epoch": math.inf,

    "evaluate_every_n_epoch": 1,
    "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

    "save_every_n_epoch": math.inf,
    "load_weights": False,
    "load_path": "result",

    "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
}

config_quad_preprocessing = {
    "config_name": "quadratic_preprocessing",
    "objective_network_generator": lambda: QuadraticFunctionLayer(16),
    "num_layers": 1,
    "objective_loss_fn": lambda y_true, y_pred: y_pred,
    "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),

    "dataset": quadratic_dataset,
    "evaluation_size": 0.5,
    "batch_size": 1,

    "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.001, dense_trainable=False),
    "one_optimizer": True,
    "optimizer_optimizer": keras.optimizers.Adam(),
    "train_optimizer_steps": 16,
    "accumulate_losses": tf.add_n,
    "train_optimizer_every_step": False,

    "super_epochs": 25,
    "epochs": 1,
    "max_steps_per_super_epoch": 320,

    "evaluate_every_n_epoch": 1,
    "evaluation_metric": QuadMetric(),

    "save_every_n_epoch": math.inf,
    "load_weights": False,
    "load_path": "result",

    "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
}

config_quad_no_preprocessing = {
    "config_name": "quadratic_no_preprocessing",
    "objective_network_generator": lambda: QuadraticFunctionLayer(16),
    "num_layers": 1,
    "objective_loss_fn": lambda y_true, y_pred: y_pred,
    "objective_gradient_preprocessor": lambda x: x,

    "dataset": quadratic_dataset,
    "evaluation_size": 0.5,
    "batch_size": 1,

    "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.001, dense_trainable=False),
    "one_optimizer": True,
    "optimizer_optimizer": keras.optimizers.Adam(),
    "train_optimizer_steps": 16,
    "accumulate_losses": tf.add_n,
    "train_optimizer_every_step": False,

    "super_epochs": 25,
    "epochs": 1,
    "max_steps_per_super_epoch": 320,

    "evaluate_every_n_epoch": 1,
    "evaluation_metric": QuadMetric(),

    "save_every_n_epoch": math.inf,
    "load_weights": False,
    "load_path": "result",

    "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
}

def transfer_mnist_to_fashion_mnist():
    plt.rcParams['text.usetex'] = True

    tf.random.set_seed(1)

    ltl_1 = LearningToLearn(config_quad_no_preprocessing)
    ltl_1.train_optimizer()
    ltl_1.evaluate_optimizer("1", label="Without Preprocessing", clear_figure=False)

    # ltl_2 = LearningToLearn(config_quad_preprocessing)
    # ltl_2.train_optimizer()
    # ltl_2.evaluate_optimizer("1", label="With Preprocessing", clear_figure=False)

def main():
    transfer_mnist_to_fashion_mnist()
    # for i in range(100):
    #     tf.random.set_seed(i)
    #     print(i)

if __name__ == "__main__":
    main()
