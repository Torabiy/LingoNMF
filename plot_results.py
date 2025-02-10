# plot_results.py
# This module generates plots for visualizing the separation results.

import matplotlib.pyplot as plt
import numpy as np

def plot_separation_results(Y, XH, X):
    """Plots original signals, mixed signals, and separated signals."""
    titles = ['Mix 1', 'Mix 2', 's_{hat} 1', 's_{hat} 2', 'Original Heart', 'Original Lung', 'Normalized s_{hat} 1', 'Normalized s_{hat} 2']
    data = [Y[0, :], Y[1, :], XH[0, :], XH[1, :], X[0, :], X[1, :], XH[0, :], XH[1, :]]
    
    plt.figure(figsize=(12, 8))
    
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        if i >= 6:  # Normalize the last two plots
            data[i] -= np.mean(data[i])  # subtract the mean
            data[i] /= np.max(np.abs(data[i]))  # divide by the maximum absolute value for normalization
        plt.plot(data[i])
        plt.title(titles[i])
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
