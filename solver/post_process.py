import matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats.distributions import chi2 as chisq


def get_chi2_test(chi2_test: tf.Tensor, dof: int, alpha: float = 0.05, v: bool = False) -> int:
    """
    TODO: write docstring
    """
    lower_bound = chisq.ppf(alpha / 2, dof)
    upper_bound = chisq.ppf(1 - (alpha / 2), dof)
    if v:
        print(f'intervals {np.array([lower_bound, upper_bound]).round(5)}, test_chi2 {chi2_test.round(5):.5f}')
    if chi2_test <= lower_bound:
        if v:
            print(f'theoretical values are too close to practical ones at alpha={alpha}')
        return -1
    elif lower_bound < chi2_test < upper_bound:
        if v:
            print(f'Hypothesis H0 holds at alpha={alpha}')
        return 0
    else:
        if v:
            print(f'Hypothesis H0 must be rejected at alpha={alpha}')
        return 1


def plot_l1_norms(norms_list: np.array, circ_nos: np.array) -> None:
    """
    TODO: write docstring
    """
    figsize = (16, 9)
    f, axarr = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = axarr

    pts = range(norms_list.shape[0])

    for idx in pts:
        ax1.plot(circ_nos, norms_list[:, idx, 0], label=f'Circuit {int(circ_nos[idx])}')
        ax2.plot(circ_nos, norms_list[:, idx, 1], label=f'Circuit {int(circ_nos[idx])}')

    ax1.set_xscale('log')
    ax1.set_xticks(np.logspace(1, 10, 10, base=2))
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax2.set_xscale('log')
    ax2.set_xticks(np.logspace(1, 10, 10, base=2))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax1.set_title('Ideal')
    ax2.set_title('True')

    ax1.legend()
    ax2.legend()
