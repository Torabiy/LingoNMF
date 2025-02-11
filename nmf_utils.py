import matplotlib.pyplot as plt
import numpy as np

def plot_results(Y, XH, X):
    """
    Plots the mixed, recovered, and original signals.
    """
    titles = ['Mix 1', 'Mix 2', 'Recovered Source 1', 'Recovered Source 2', 'Original Heart', 'Original Lung', 'Normalized Source 1', 'Normalized Source 2']
    data = [Y[0, :], Y[1, :], XH[0, :], XH[1, :], X[1, :], X[0, :], XH[0, :], XH[1, :]]

    plt.figure(figsize=(12, 8))
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        if i >= 6:
            data[i] -= np.mean(data[i])
            data[i] /= np.max(np.abs(data[i]))
        plt.plot(data[i])
        plt.title(titles[i])
        plt.grid(True)

    plt.tight_layout()
    plt.show()
