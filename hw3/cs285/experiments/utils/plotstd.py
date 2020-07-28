import numpy as np
from matplotlib import pyplot as plt

def plot_mean_std(ax, mean, std=None, x=None, label=None, c='y', linestyle='-', **kwargs):
    """Plot error bar for standard deviation
    """

    if x is None: 
        x = np.arange(len(mean))

    ax.plot(x, mean, label=label, c=c, linestyle=linestyle, **kwargs)
    if std is not None:
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, facecolor=c)