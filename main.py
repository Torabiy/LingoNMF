# Copyright (c) 2023 Y. Torabi et al.
# Cite as: Y. Torabi et al., "A New Non-Negative Matrix Factorization Approach
# for Blind Source Separation of Cardiovascular and Respiratory Sound Based
# on the Periodicity of Heart and Lung Function," arXiv preprint,
# arXiv:2305.01889, 2023.

# main.py
# This script orchestrates the data loading, processing, and separation using NMF.

import numpy as np
from data_loader import load_audio_data
from nmf_separation import perform_nmf
from plot_results import plot_separation_results

# Load data
y1, fs1, y2, fs2 = load_audio_data('M_AF_LC.wav', 'M_W_RLA.wav')

# Ensure signals are of equal length
min_length = min(len(y1), len(y2))
h = y1[:min_length]
l = y2[:min_length]

# Stack signals into a matrix
X = np.vstack([h, l])

# Define mixing matrix
np.random.seed(100)
A = np.random.rand(2, 2)  # Random mixing matrix

# Generate mixtures
Y = 5 * np.dot(A, X) + 6

# NMF options
options = {'J': 2, 'niter': 1000}

# Perform NMF and evaluate separation quality
H, XH = perform_nmf(Y, options, X, fs1)

# Plot results
plot_separation_results(Y, XH, X)
