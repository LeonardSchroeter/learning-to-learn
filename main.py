import math

import tensorflow as tf
from tensorflow import keras

from experiments.mnist_large_number_steps import mnist_large
from experiments.mnist_preprocessing import mnist_preprocessing
from experiments.mnist_preprocessing_optimizer import \
    mnist_preprocessing_optimizer
from experiments.mnist_problems import mnist_problems
from experiments.mnist_unroll import mnist_unroll
from experiments.quadratic_large_number_steps import quadratic_large
from experiments.quadratic_preprocessing import quadratic_preprocessing
from experiments.quadratic_preprocessing_optimizer import \
    quadratic_preprocessing_optimizer
from experiments.quadratic_unroll import quadratic_unroll
from experiments.quadratic_weights import quadratic_weights

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
# preprocess_optimizer_gradients - boolean - if true, preprocess the optimizer gradients - default: True
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
    "preprocess_optimizer_gradients": True,
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

def main():
    # experiments for the thesis
    # resulting plots are saved in the "plots" folder which needs to exists
    # section 5.2
    quadratic_large()
    # section 5.2
    mnist_large()
    # section 5.3.1
    quadratic_preprocessing()
    # section 5.3.1
    mnist_preprocessing()
    # section 5.3.2
    quadratic_preprocessing_optimizer()
    # section 5.3.2
    mnist_preprocessing_optimizer()
    # all tests from subsections of section 5.4
    mnist_problems()
    # section 5.5.2
    quadratic_unroll()
    # section 5.5.2
    mnist_unroll()
    # section 5.6
    quadratic_weights()

if __name__ == "__main__":
    main()
