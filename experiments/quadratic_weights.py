import tensorflow as tf
from src.custom_metrics import QuadMetric
from src.objectives import QuadraticFunctionLayer
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.util import weighted_sum
from tensorflow import keras

from experiments.util import experiment


def quadratic_weights():
    quadratic_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([640]), tf.zeros([640])))
    W = tf.random.normal([16, 16])
    y = tf.random.normal([16])

    quadratic_weights_1_every = {
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
    }

    quadratic_weights_inc_every = {
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
        "accumulate_losses": lambda losses: weighted_sum(losses, 1.0),
        "train_optimizer_every_step": False,

        "super_epochs": 25,
        "epochs": 1,
        "max_steps_per_super_epoch": 320,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": QuadMetric(),
    }

    quadratic_weights_1_once = {
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
        "train_optimizer_steps": 80,
        "accumulate_losses": tf.add_n,
        "train_optimizer_every_step": False,

        "super_epochs": 100,
        "epochs": 1,
        "max_steps_per_super_epoch": 80,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": QuadMetric(),
    }

    quadratic_weights_inc_once = {
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
        "train_optimizer_steps": 80,
        "accumulate_losses": lambda losses: weighted_sum(losses, 1.0),
        "train_optimizer_every_step": False,

        "super_epochs": 100,
        "epochs": 1,
        "max_steps_per_super_epoch": 80,

        "evaluate_every_n_epoch": 1,
        "evaluation_metric": QuadMetric(),
    }

    experiment([quadratic_weights_1_every, quadratic_weights_inc_every], ["$w_t = 1$", "$w_t = \\beta t + 1$"], 10, "quadratic_weights_every")
    experiment([quadratic_weights_1_once, quadratic_weights_inc_once], ["$w_t = 1$", "$w_t = \\beta t + 1$"], 10, "quadratic_weights_once")
