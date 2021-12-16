import math

import matplotlib.pyplot as plt
import tensorflow as tf
from src.custom_metrics import QuadMetric
from src.objectives import MLP, ConvNN, QuadraticFunctionLayer
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.train import LearningToLearn
from src.util import preprocess_gradients
from tensorflow import keras


def quadratic_preprocessing():
    quadratic_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([640]), tf.zeros([640])))

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

    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    plt.subplots_adjust(bottom=0.15, top=0.9)

    tf.random.set_seed(1)

    ltl_1 = LearningToLearn(config_quad_no_preprocessing)
    ltl_1.train_optimizer()
    ltl_1.evaluate_optimizer("1", label="Without Preprocessing", clear_figure=False)

    ltl_2 = LearningToLearn(config_quad_preprocessing)
    ltl_2.train_optimizer()
    ltl_2.evaluate_optimizer("1", label="With Preprocessing")
