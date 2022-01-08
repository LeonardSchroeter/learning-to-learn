import math

import matplotlib.pyplot as plt
import tensorflow as tf
from src.custom_metrics import QuadMetric
from src.objectives import (MLP, ConvNN, MLPLeakyRelu, MLPRelu, MLPSigmoid,
                            MLPTanh, QuadraticFunctionLayer)
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.train import LearningToLearn
from src.util import preprocess_gradients
from tensorflow import keras


def quadratic_large():
    quadratic_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([640]), tf.zeros([640])))
    quadratic_dataset_large = tf.data.Dataset.from_tensor_slices((tf.zeros([6400]), tf.zeros([6400])))

    W = tf.random.normal([16, 16])
    y = tf.random.normal([16])

    quadratic_normal = {
        "config_name": "quadratic_normal",
        "objective_network_generator": lambda same: QuadraticFunctionLayer(16, same, W, y),
        "num_layers": 1,
        "objective_loss_fn": lambda y_true, y_pred: y_pred,
        "objective_gradient_preprocessor": lambda x: x,

        "dataset": quadratic_dataset,
        "evaluation_size": 0.5,
        "batch_size": 1,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.001, dense_trainable=False),
        "one_optimizer": True,
        "preprocess_optimizer_gradients": True,
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

    }

    quadratic_large = {
        "config_name": "quadratic_large",
        "objective_network_generator": lambda same: QuadraticFunctionLayer(16, same, W, y),
        "num_layers": 1,
        "objective_loss_fn": lambda y_true, y_pred: y_pred,
        "objective_gradient_preprocessor": lambda x: x,

        "dataset": quadratic_dataset_large,
        "evaluation_size": 0.5,
        "batch_size": 1,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.001, dense_trainable=False),
        "one_optimizer": True,
        "preprocess_optimizer_gradients": True,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 25,
        "epochs": 1,
        "max_steps_per_super_epoch": 3200,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": QuadMetric(),

        "save_every_n_epoch": math.inf,
        "load_weights": False,
        "load_path": "result",

    }

    tf.random.set_seed(1)

    ltl_2 = LearningToLearn(quadratic_normal)
    ltl_2.train_optimizer()
    _, weights = ltl_2.evaluate_optimizer("test", label="320 steps", clear_figure=False)

    ltl_1 = LearningToLearn(quadratic_large)
    ltl_1.train_optimizer()

    ltl_1.dataset = quadratic_dataset
    ltl_1.max_steps_per_super_epoch = 320

    _, weights = ltl_1.evaluate_optimizer("test", label="3200 steps", objective_network_weights=weights)
