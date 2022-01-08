import matplotlib.pyplot as plt
import numpy as np

from src.plot import hist, plot


def main():
    pass
    # losses = np.loadtxt("tmp/mnist_unroll_8_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="$T$ = 8", filename="mnist_unroll_8", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_unroll_16_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="$T$ = 16", filename="mnist_unroll_16", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_unroll_32_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="$T$ = 32", filename="mnist_unroll_32")

    # losses = np.loadtxt("tmp/mnist_no_pre_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="Without Preprocessing", filename="mnist_no_preprocessing", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_pre_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="With Preprocessing", filename="mnist_preprocessing")

    # losses = np.loadtxt("tmp/mnist_no_pre_opt_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="Without Preprocessing", filename="mnist_no_preprocessing_optimizer", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_pre_opt_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="With Preprocessing", filename="mnist_preprocessing_optimizer")

    # losses = np.loadtxt("tmp/mnist_different_sigmoid_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="Sigmoid", filename="mnist_different_sigmoid", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_different_tanh_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="Tanh", filename="mnist_different_tanh")

    # losses = np.loadtxt("tmp/mnist_different_relu_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="ReLU", filename="mnist_different_relu", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_different_leaky_relu_losses.csv", delimiter=",")
    # losses = losses[:-1]
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="Leaky ReLU", filename="mnist_different_leaky_relu")

    # losses1 = np.loadtxt("tmp/mnist_different_relu_losses.csv", delimiter=",")
    # losses1 = [loss[-1] for loss in losses1]

    # losses2 = np.loadtxt("tmp/mnist_different_leaky_relu_losses.csv", delimiter=",")
    # losses2 = [loss[-1] for loss in losses2]

    # losses3 = np.loadtxt("tmp/mnist_different_sigmoid_losses.csv", delimiter=",")
    # losses3 = [loss[-1] for loss in losses3]

    # losses4 = np.loadtxt("tmp/mnist_different_tanh_losses.csv", delimiter=",")
    # losses4 = [loss[-1] for loss in losses4]


    # hist([losses1, losses2], label=["ReLU", "Leaky ReLU"], filename="mnist_different_relu_leaky_relu")
    # hist([losses3, losses4], label=["Sigmoid", "Tanh"], filename="mnist_different_sigmoid_tanh")

    # hist([losses1, losses2, losses3, losses4], label=["ReLU", "Leaky ReLU", "Sigmoid", "Tanh"], filename="mnist_different_relu_leaky_relu_sigmoid_tanh")

    # losses = np.loadtxt("tmp/mnist_one_relu_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="One Optimizer", filename="mnist_one_relu", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_mul_relu_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="Multiple Optimizers", filename="mnist_mul_relu")

    # losses = np.loadtxt("tmp/mnist_one_sigmoid_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="One Optimizer", filename="mnist_one_sigmoid", clear_figure=False)

    # losses = np.loadtxt("tmp/mnist_mul_sigmoid_losses.csv", delimiter=",")
    # loss = losses[losses[:, -1].argmin()]
    # plot(loss, label="Multiple Optimizers", filename="mnist_mul_sigmoid")

    # losses1 = np.loadtxt("tmp/mnist_one_relu_losses.csv", delimiter=",")
    # losses1 = [loss[-1] for loss in losses1]

    # losses2 = np.loadtxt("tmp/mnist_mul_relu_losses.csv", delimiter=",")
    # losses2 = [loss[-1] for loss in losses2]

    # losses3 = np.loadtxt("tmp/mnist_one_sigmoid_losses.csv", delimiter=",")
    # losses3 = [loss[-1] for loss in losses3]

    # losses4 = np.loadtxt("tmp/mnist_mul_sigmoid_losses.csv", delimiter=",")
    # losses4 = [loss[-1] for loss in losses4]


    # hist([losses1, losses2], label=["One Optimizer", "Multiple Optimizers"], filename="mnist_comp_relu")
    # hist([losses3, losses4], label=["One Optimizer", "Multiple Optimizers"], filename="mnist_comp_sigmoid")

    # hist([losses1, losses2, losses3, losses4], label=["ReLU", "Leaky ReLU", "Sigmoid", "Tanh"], filename="asdfawef")

    # losses1 = np.loadtxt("tmp/mnist_sigmoid_default.csv", delimiter=",")
    # losses1 = [loss[-1] for loss in losses1]

    # losses2 = np.loadtxt("tmp/mnist_sigmoid_pre.csv", delimiter=",")
    # losses2 = [loss[-1] for loss in losses2]

    # losses3 = np.loadtxt("tmp/mnist_relu_default.csv", delimiter=",")
    # losses3 = [loss[-1] for loss in losses3]

    # losses4 = np.loadtxt("tmp/mnist_relu_pre.csv", delimiter=",")
    # losses4 = [loss[-1] for loss in losses4]


    # hist([losses1, losses2], label=["Without Pretraining", "With Pretraining"], filename="mnist_sigmoid_pre")
    # hist([losses3, losses4], label=["Without Pretraining", "With Pretraining"], filename="mnist_relu_pre")

    # losses1 = np.loadtxt("tmp/mnist_sigmoid_default.csv", delimiter=",")
    # losses1 = [loss[-1] for loss in losses1]

    # losses2 = np.loadtxt("tmp/mnist_tanh_default.csv", delimiter=",")
    # losses2 = [loss[-1] for loss in losses2]

    # losses3 = np.loadtxt("tmp/mnist_relu_default.csv", delimiter=",")
    # losses3 = [loss[-1] for loss in losses3]

    # losses4 = np.loadtxt("tmp/mnist_leaky_relu_default.csv", delimiter=",")
    # losses4 = [loss[-1] for loss in losses4]


    # hist([losses1, losses2], label=["Sigmoid", "Tanh"], filename="mnist_sigmoid_tanh")
    # hist([losses3, losses4], label=["ReLU", "Leaky ReLU"], filename="mnist_relu_leaky_relu")

    losses1 = np.loadtxt("tmp/mnist_sigmoid_default.csv", delimiter=",")
    losses1 = [loss[-1] for loss in losses1]

    losses2 = np.loadtxt("tmp/mnist_mul_sigmoid.csv", delimiter=",")
    losses2 = [loss[-1] for loss in losses2]

    losses3 = np.loadtxt("tmp/mnist_relu_default.csv", delimiter=",")
    losses3 = [loss[-1] for loss in losses3]

    losses4 = np.loadtxt("tmp/mnist_mul_relu.csv", delimiter=",")
    losses4 = [loss[-1] for loss in losses4]


    hist([losses1, losses2], label=["One Optimizer", "Multiple Optimizers"], filename="mnist_sigmoid_mul")
    hist([losses3, losses4], label=["One Optimizer", "Multiple Optimizers"], filename="mnist_relu_mul")

if __name__ == "__main__":
    main()
