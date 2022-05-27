import tensorflow as tf
from src.custom_metrics import QuadMetric
from src.objectives import QuadraticFunctionLayer
from src.optimizer_rnn import LSTMNetworkPerParameter
from tensorflow import keras

from experiments.util import experiment


def quadratic_preprocessing_optimizer():
    quadratic_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([640]), tf.zeros([640])))

    W = tf.random.normal([16, 16])
    y = tf.random.normal([16])

    config_quad_preprocessing_optimizer = {
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

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    config_quad_preprocessing_no_optimizer = {
        "objective_network_generator": lambda same: QuadraticFunctionLayer(16, same, W, y),
        "num_layers": 1,
        "objective_loss_fn": lambda y_true, y_pred: y_pred,
        "objective_gradient_preprocessor": lambda x: x,

        "dataset": quadratic_dataset,
        "evaluation_size": 0.5,
        "batch_size": 1,

        "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.001, dense_trainable=False),
        "one_optimizer": True,
        "preprocess_optimizer_gradients": False,
        "optimizer_optimizer": keras.optimizers.Adam(),
        "train_optimizer_steps": 16,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 25,
        "epochs": 1,
        "max_steps_per_super_epoch": 320,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": QuadMetric(),

        "comparison_optimizers": [keras.optimizers.SGD(), keras.optimizers.Adam()],
    }

    experiment([config_quad_preprocessing_no_optimizer, config_quad_preprocessing_optimizer], ["Without Preprocessing", "With Preprocessing"], 10, "quadratic_preprocessing_optimizer")
