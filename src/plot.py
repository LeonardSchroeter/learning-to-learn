import matplotlib.pyplot as plt
import numpy as np


# Plots the loss for a single run
# Used for all plots in the thesis
def plot(losses, label, filename, clear_figure=True):
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    plt.subplots_adjust(bottom=0.15, top=0.9)

    x = list(range(1, len(losses) + 1))

    plt.plot(x, losses, label=label)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    plt.savefig(f"plots/{filename}.eps")

    if clear_figure:
        plt.clf()

# Plots the histograms for the last losses of multiple runs
# Used for all histograms in the thesis
def hist(losses, label, filename, clear_figure=True):
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    plt.subplots_adjust(bottom=0.15, top=0.9)

    bins = np.arange(0, 3.5, 0.5)

    x = [np.clip(loss, bins[0], bins[-1]) for loss in losses]
    plt.hist(x, bins=bins, label=label)

    xlabels = bins.astype(str)
    xlabels[-1] = "$\infty$"
    plt.xticks(bins, xlabels)
    plt.yticks([])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))

    plt.savefig(f"plots/{filename}.eps")

    if clear_figure:
        plt.clf()
