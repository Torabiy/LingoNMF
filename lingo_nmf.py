"""
Copyright © 2025 Yasaman Torabi

Cite this work as:
Torabi, Yasaman; Shirani, Shahram; Reilly, James P. (2025), 
Large Language Model-based Nonnegative Matrix Factorization For Cardiorespiratory Sound Separation, 
arXiv preprint, https://doi.org/10.48550/arXiv.2502.05757.
"""
import numpy as np
from scipy.signal import periodogram

def estimate_f0(signal, fs, f_min=0.1, f_max=100.0):
    """
    Estimates the fundamental frequency of a signal.
    """
    f, Pxx = periodogram(signal, fs=fs)
    valid_mask = (f >= f_min) & (f <= f_max)
    return f[valid_mask][np.argmax(Pxx[valid_mask])] if np.any(valid_mask) else None

def lingonmf(Y, options, fs):
    """
    Lingo NMF: Implements NMF with a penalty for fundamental frequency.
    """
    np.random.seed(42)
    J, T = Y.shape
    W = np.random.randn(J, options['J']) + 0.5
    H = np.random.randn(options['J'], T) + 10

    target_f0 = np.array([0.5, 0.05])  # Example values for heart & lung
    lambda_f0 = np.array([0.001, 0.001])
    alpha = 0.5
    
    for _ in range(options['niter']):
        WH = W @ H + 1e-10
        f0_weights = np.array([estimate_f0(H[i, :], fs) for i in range(H.shape[0])])
        f0_weights = np.nan_to_num(f0_weights, nan=0)
        f0_penalty = np.sum(lambda_f0 * (f0_weights - target_f0) ** 2)
        W *= (Y @ H.T) / (W @ (H @ H.T) + f0_penalty) ** alpha
        H *= (W.T @ Y) / ((W.T @ W) @ H + f0_penalty)

    return W, H, f0_penalty
