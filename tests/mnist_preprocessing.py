import math

import matplotlib.pyplot as plt
import tensorflow as tf
from src.custom_metrics import QuadMetric
from src.objectives import MLP, ConvNN, QuadraticFunctionLayer
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.train import LearningToLearn
from src.util import preprocess_gradients
from tensorflow import keras


def mnist_preprocessing():
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    mnist_dnn_dataset = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255, y_train))

    mnist_preprocessing = {
        "config_name": "mnist_preprocessing",
        "objective_network_generator": lambda: MLP(),
        "num_layers": 2,
        "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
        "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),

        "dataset": mnist_dnn_dataset,
        "evaluation_size": 0.2,
        "batch_size": 128,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01, dense_trainable=False),
        "one_optimizer": True,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 25,
        "epochs": 1,
        "max_steps_per_super_epoch": math.inf,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }


    mnist_no_preprocessing = {
        "config_name": "mnist_no_preprocessing",
        "objective_network_generator": lambda: MLP(),
        "num_layers": 2,
        "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
        "objective_gradient_preprocessor": lambda x: x,

        "dataset": mnist_dnn_dataset,
        "evaluation_size": 0.2,
        "batch_size": 128,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01, dense_trainable=False),
        "one_optimizer": True,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 25,
        "epochs": 1,
        "max_steps_per_super_epoch": math.inf,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    plt.subplots_adjust(bottom=0.15, top=0.9)

    tf.random.set_seed(4)

    ltl_2 = LearningToLearn(mnist_no_preprocessing)
    ltl_2.train_optimizer()
    ltl_2.evaluate_optimizer("1", label="Without Preprocessing", clear_figure=False)

    ltl_1 = LearningToLearn(mnist_preprocessing)
    ltl_1.train_optimizer()
    ltl_1.evaluate_optimizer("1", label="With Preprocessing")
