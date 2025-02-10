# lingo_nmf.py
# Copyright (c) 2025 Y. Torabi et al.
# Cite as: https://doi.org/10.48550/arXiv.2305.01889

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

def nmf_with_f0_penalty(Y, options, X, fs):
    """
    Implements NMF with an added penalty for fundamental frequency deviation.
    """

    # Initialize matrices (random non-negative values)
    np.random.seed(42)
    J, T = Y.shape
    W = np.random.randn(J, options['J']) + 1  # Basis matrix
    H = np.random.randn(options['J'], T) + 10  # Coefficients matrix

    # Define target fundamental frequencies (for heart and lung sounds)
    target_f0 = np.array([0.5, 0.05])  # Hz (example values for heart & lung)
    lambda_f0 = np.array([0.001, 0.001])  # Different penalties for heart & lung
    
    # Perform Iterative Multiplicative Updates with F0 Penalty
    for _ in range(options['niter']):
        WH = W @ H + 1e-10  # Avoid division by zero

        # Compute the fundamental frequency for each source
        f0_weights = np.array([estimate_f0(H[i, :], fs) for i in range(H.shape[0])])
        f0_weights = np.nan_to_num(f0_weights, nan=0)  # Handle NaNs

        # Compute F0 penalty term (difference squared)
        f0_penalty = np.sum(lambda_f0 * (f0_weights - target_f0) ** 2)

        # Update W and H with the cost function including F0 penalty
        W *= (Y @ H.T) / (W @ (H @ H.T) + f0_penalty)
        H *= (W.T @ Y) / ((W.T @ W) @ H + f0_penalty)

    return W, H, f0_penalty

def estimate_f0(signal, fs, f_min=0.1, f_max=100.0):
    """
    Estimates the fundamental frequency of a signal.
    """
    f, Pxx = periodogram(signal, fs=fs)
    valid_mask = (f >= f_min) & (f <= f_max)  # Restrict to plausible frequencies
    return f[valid_mask][np.argmax(Pxx[valid_mask])] if np.any(valid_mask) else None
