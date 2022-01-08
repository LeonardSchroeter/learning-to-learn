import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.custom_metrics import QuadMetric
from src.objectives import (MLP, ConvNN, MLPLeakyRelu, MLPRelu, MLPSigmoid,
                            MLPTanh, QuadraticFunctionLayer)
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.train import LearningToLearn
from src.util import preprocess_gradients
from tensorflow import keras


def mnist_different_activations():
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    mnist_dnn_dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255, y_train))

    mnist_sigmoid = {
        "config_name": "mnist_sigmoid",
        "objective_network_generator": lambda _: MLPSigmoid(),
        "num_layers": 2,
        "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
        "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),

        "dataset": mnist_dnn_dataset,
        "evaluation_size": 0.2,
        "batch_size": 128,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01, dense_trainable=False),
        "one_optimizer": True,
        "preprocess_optimizer_gradients": True,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 10,
        "epochs": 1,
        "max_steps_per_super_epoch": math.inf,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    mnist_relu = {
        "config_name": "mnist_relu",
        "objective_network_generator": lambda _: MLPRelu(),
        "num_layers": 2,
        "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
        "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),

        "dataset": mnist_dnn_dataset,
        "evaluation_size": 0.2,
        "batch_size": 128,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01, dense_trainable=False),
        "one_optimizer": True,
        "preprocess_optimizer_gradients": True,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 10,
        "epochs": 1,
        "max_steps_per_super_epoch": math.inf,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    mnist_leaky_relu = {
        "config_name": "mnist_leaky_relu",
        "objective_network_generator": lambda _: MLPLeakyRelu(),
        "num_layers": 2,
        "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
        "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),

        "dataset": mnist_dnn_dataset,
        "evaluation_size": 0.2,
        "batch_size": 128,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01, dense_trainable=False),
        "one_optimizer": True,
        "preprocess_optimizer_gradients": True,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 5,
        "epochs": 1,
        "max_steps_per_super_epoch": 160,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    mnist_tanh = {
        "config_name": "mnist_tanh",
        "objective_network_generator": lambda _: MLPTanh(),
        "num_layers": 2,
        "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
        "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),

        "dataset": mnist_dnn_dataset,
        "evaluation_size": 0.2,
        "batch_size": 128,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01, dense_trainable=False),
        "one_optimizer": True,
        "preprocess_optimizer_gradients": True,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 5,
        "epochs": 1,
        "max_steps_per_super_epoch": 160,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    # losses = []
    w = None

    # for i in range(10):
    #     tf.random.set_seed(i)

    #     ltl = LearningToLearn(mnist_sigmoid)
    #     ltl.train_optimizer()
    #     l, w = ltl.evaluate_optimizer("test", label="Sigmoid", objective_network_weights=w)

    #     losses.append(l)

    # np.savetxt("tmp/mnist_different_sigmoid_losses.csv", losses, delimiter=",")

    losses = []
    
    for i in range(100):
        tf.random.set_seed(i)

        ltl = LearningToLearn(mnist_tanh)
        ltl.train_optimizer()
        l, w = ltl.evaluate_optimizer("test", label="Tanh", objective_network_weights=w)

        losses.append(l)

    np.savetxt("tmp/mnist_tanh_default.csv", losses, delimiter=",")

    # losses = []

    # for i in range(10):
    #     tf.random.set_seed(i)

    #     ltl = LearningToLearn(mnist_relu)
    #     ltl.train_optimizer()
    #     l, w = ltl.evaluate_optimizer("test", label="Relu", objective_network_weights=w)

    #     losses.append(l)

    # np.savetxt("tmp/mnist_different_relu_losses.csv", losses, delimiter=",")

    losses = []

    for i in range(100):
        tf.random.set_seed(i)

        ltl = LearningToLearn(mnist_leaky_relu)
        ltl.train_optimizer()
        l, w = ltl.evaluate_optimizer("test", label="Leaky Relu", objective_network_weights=w)

        losses.append(l)

    np.savetxt("tmp/mnist_leaky_relu_default.csv", losses, delimiter=",")
