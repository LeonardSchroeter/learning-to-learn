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


def mnist_unroll():
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    mnist_dnn_dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255, y_train))

    mnist_unroll_8 = {
        "config_name": "mnist_unroll_8",
        "objective_network_generator": lambda _: MLP(),
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
        "train_optimizer_steps": 8,
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
    }

    mnist_unroll_16 = {
        "config_name": "mnist_unroll_16",
        "objective_network_generator": lambda _: MLP(),
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
    }

    mnist_unroll_32 = {
        "config_name": "mnist_unroll_32",
        "objective_network_generator": lambda _: MLP(),
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
        "train_optimizer_steps": 32,
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
    }

    losses = []
    w = None

    for i in range(10):
        tf.random.set_seed(i)

        ltl = LearningToLearn(mnist_unroll_8)
        ltl.train_optimizer()
        l, w = ltl.evaluate_optimizer("0", label="$T$ = 8", objective_network_weights=w)

        losses.append(l)

    np.savetxt("tmp/mnist_unroll_8_losses.csv", losses, delimiter=",")

    losses = []
    
    for i in range(10):
        tf.random.set_seed(i)

        ltl = LearningToLearn(mnist_unroll_16)
        ltl.train_optimizer()
        l, w = ltl.evaluate_optimizer("0", label="$T$ = 16", objective_network_weights=w)

        losses.append(l)

    np.savetxt("tmp/mnist_unroll_16_losses.csv", losses, delimiter=",")

    losses = []

    for i in range(10):
        tf.random.set_seed(i)

        ltl = LearningToLearn(mnist_unroll_32)
        ltl.train_optimizer()
        l, w = ltl.evaluate_optimizer("0", label="$T$ = 32", objective_network_weights=w)

        losses.append(l)

    np.savetxt("tmp/mnist_unroll_32_losses.csv", losses, delimiter=",")
