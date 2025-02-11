"""
Copyright © 2025 Yasaman Torabi, Shahram Shirani, James P. Reilly

Cite this work as:
Torabi, Yasaman; Shirani, Shahram; Reilly, James P. (2025), 
Large Language Model-based Nonnegative Matrix Factorization For Cardiorespiratory Sound Separation, 
arXiv preprint, https://doi.org/10.48550/arXiv.2502.05757.
"""
import numpy as np
from dataset_loader import load_audio_files
from standard_nmf import standard_nmf
from alpha_nmf import alpha_nmf
from pl_nmf import pl_nmf
from lingonmf import lingonmf
from evaluation import evaluate_separation
from nmf_utils import plot_results
from config import options

# Load dataset
X, fs = load_audio_files()

# Assume a predefined mixing matrix A
A = np.array([[0.6, 0.4], [0.4, 0.6]])

# Generate mixed signals
Y = 5 * np.dot(A, X) + 6

# Apply different NMF techniques
H_std, XH_std = standard_nmf(Y, options)
H_alpha, XH_alpha = alpha_nmf(Y, options)
W_lingo, H_lingo, f0_penalty = lingonmf(Y, options, fs)

# Evaluate separation
sdr, sir, sar = evaluate_separation(X, XH_std)

# Plot results
plot_results(Y, XH_std, X)

# Print results
print("SIR:", sir)
print("SDR:", sdr)
print("SAR:", sar)
