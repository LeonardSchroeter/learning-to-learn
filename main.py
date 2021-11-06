import math

import tensorflow as tf
from tensorflow import keras

from src.objectives import ConvNN
from src.optimizer_rnn import LSTMNetworkPerParameter
from src.train import LearningToLearn

################################################################################
# config parameters
################################################################################
# config_name - name of the config - required
# objective_network_generator - callable that returns a network for training the objective - required
# objective_loss_fn - a tensorflow loss function - required
# objective_gradient_preprocessor - callable that preprocesses the objective gradient - default: lambda x: x

# dataset - dataset - required
# evaluation_size - fraction of dataset to use for evaluation - default 0.2
# batch_size - size of batches for training - default: 64

# optimizer_network_generator - callable that returns a network for training the optimizer - required
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
    "objective_loss_fn": None,
    "objective_gradient_preprocessor": lambda x: x,

    "dataset": None,
    "evaluation_size": 0.2,
    "batch_size": 64,

    "optimizer_network_generator": None,
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

quadratic_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([2000]), tf.zeros([2000])))

config_mnist = {
    "config_name": "mnist",
    "objective_network_generator": lambda: ConvNN(),
    "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
    "objective_gradient_preprocessor": lambda x: x,

    "dataset": mnist_dataset,
    "evaluation_size": 0.2,
    "batch_size": 64,

    "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.01),
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

    "comparison_optimizers": [],
}

# config_mnist_2 = {
#     "config_name": "mnist_2",
#     "dataset": mnist_dataset,
#     "batch_size": 64,
#     "evaluation_size": 0.2,
#     "optimizer_network_generator": lambda: LSTMNetworkPerParameter(0.1, False),
#     "optimizer_optimizer": keras.optimizers.Adam(),
#     "train_optimizer_steps": 16,
#     "train_optimizer_every_step": False,
#     "objective_network_generator": lambda: ConvNN(),
#     "objective_loss_fn": keras.losses.SparseCategoricalCrossentropy(),
#     "accumulate_losses": tf.add_n,
#     "evaluation_metric": keras.metrics.SparseCategoricalAccuracy(),
#     "super_epochs": 50,
#     "epochs": 10,
#     "comparison_optimizers": [tf.keras.optimizers.Adam()],
#     # "objective_gradient_preprocessor": lambda x: preprocess_gradients(x, 10),
#     "max_steps_per_super_epoch": 100,
# }

def main():
    for i in range(100):
        print("Seed: ", i)
        tf.random.set_seed(i)
    
        ltl = LearningToLearn(config_mnist)
        ltl.train_optimizer()
        ltl.evaluate_optimizer()

if __name__ == "__main__":
    main()
