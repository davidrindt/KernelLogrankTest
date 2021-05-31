import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from kernel_logrank.utils.generate_synthetic_data import generate_synthetic_data

rcParams.update({'figure.autolayout': True})
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 35
plt.rcParams['axes.titlesize'] = 35
plt.rcParams['font.size'] = 35
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = "\\usepackage{subdepth}"


def survival_scatter_plot(x, z, d, filename=None, save=False):
    select = np.where(d == 1)
    unselect = np.where(d == 0)
    plt.scatter(x[select], z[select], c='navy', label='observed')
    plt.scatter(x[unselect], z[unselect], facecolors='none', edgecolors='r', label='censored')
    plt.xlabel('covariate')
    plt.ylabel('time')
    plt.gcf().subplots_adjust(left=0.138, bottom=0.175)
    if save:
        plt.savefig(filename)
    plt.show()
    return None


if __name__ == '__main__':
    # Generate some data
    X, z, d = generate_synthetic_data()

    # Make the plot
    survival_scatter_plot(X, z, d)
