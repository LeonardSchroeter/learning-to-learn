import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.plot import hist, plot
from src.train import LearningToLearn


# learn an optimizer given a config n times and return the losses of the best optimizer and the last losses
# uses the same initial weights during evaluation, these weights can also be passed in as a parameter w
def train_multiple(config, n, w = None, save = False, name = "default", pretrain = False):
    losses = []

    for i in range(n):
        tf.random.set_seed(i)

        ltl = LearningToLearn(config)
        if pretrain:
            ltl.pretrain(200_000)
        ltl.train_optimizer()
        l, w = ltl.evaluate_optimizer(objective_network_weights=w)

        losses.append(l)

    if save:
        np.savetxt(f"tmp/{name}.csv", losses, delimiter=",")

    losses = np.array(losses)
    
    best = losses[losses[:, -1].argmin()]

    last = [loss[-1] for loss in losses]

    return best, last, w

# a single experiment of running multiple configs n times and plotting either loss of each best optimizer
# or the distribution of last losses of all optimizers
def experiment(configs, labels, n, filename, pl0t = True, pretrain = []):
    if not pretrain:
        pretrain = [False] * len(configs)

    w = None
    last_losses = []
    for cfg, label, pre in zip(configs, labels, pretrain):
        losses, last, w = train_multiple(cfg, n, w, pretrain=pre)
        last_losses.append(last)
        if pl0t:
            plot(losses, label=label, filename=filename, clear_figure=False)
    
    if not pl0t:
        hist(last_losses, label=labels, filename=filename)

    plt.clf()
