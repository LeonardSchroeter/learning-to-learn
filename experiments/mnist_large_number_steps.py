import math

import tensorflow as tf
from src.objectives import MLP
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.plot import plot
from src.train import LearningToLearn
from src.util import preprocess_gradients
from tensorflow import keras


def mnist_large():
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    mnist_dnn_dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255, y_train))

    mnist_large = {
        "config_name": "mnist_large",
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

        "super_epochs": 1,
        "epochs": 10,
        "max_steps_per_super_epoch": math.inf,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    mnist_def = {
        "config_name": "mnist_def",
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

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    best1 = [math.inf]
    bestltl = None
    weights = None
    for i in range(10):
        tf.random.set_seed(i)
        ltl = LearningToLearn(mnist_def)
        ltl.train_optimizer()
        losses, weights = ltl.evaluate_optimizer(objective_network_weights=weights)
        if losses[-1] < best1[-1]:
            best1 = losses
            bestltl = ltl
    
    best2 = [math.inf]
    bestltl2 = None
    for i in range(10):
        tf.random.set_seed(i)
        ltl = LearningToLearn(mnist_large)
        ltl.train_optimizer()
        losses, weights = ltl.evaluate_optimizer(objective_network_weights=weights)
        if losses[-1] < best2[-1]:
            best2 = losses
            bestltl2 = ltl

    losses, weights = bestltl.evaluate_optimizer()
    plot(losses, label="1 epoch", filename="tmp", clear_figure=False)

    bestltl2.epochs = 1
    losses, weights = bestltl2.evaluate_optimizer(objective_network_weights=weights)
    plot(losses, label="10 epochs", filename="mnist_large_number_steps_1")

    bestltl.epochs = 10
    losses, weights = bestltl.evaluate_optimizer(objective_network_weights=weights)
    plot(losses, label="1 epoch", filename="tmp", clear_figure=False)

    bestltl2.epochs = 10
    losses, weights = bestltl2.evaluate_optimizer(objective_network_weights=weights)
    plot(losses, label="10 epochs", filename="mnist_large_number_steps_2")
