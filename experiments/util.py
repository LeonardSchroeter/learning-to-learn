import numpy as np
import tensorflow as tf
from src.train import LearningToLearn


def train_multiple(config, n, save = False, name = "default"):
    losses = []

    for i in range(n):
        tf.random.set_seed(i)

        ltl = LearningToLearn(config)
        ltl.train_optimizer()
        l, w = ltl.evaluate_optimizer("tmp", objective_network_weights=w)

        losses.append(l)

    if save:
        np.savetxt(f"tmp/{name}.csv", losses, delimiter=",")

    best = losses[losses[:, -1].argmin()]

    last = [loss[-1] for loss in losses]

    return best, last
