"""
Copyright © 2025 Yasaman Torabi, Shahram Shirani, James P. Reilly

Cite this work as:
Torabi, Yasaman; Shirani, Shahram; Reilly, James P. (2025), 
Large Language Model-based Nonnegative Matrix Factorization For Cardiorespiratory Sound Separation, 
arXiv preprint, https://doi.org/10.48550/arXiv.2502.05757.
"""
import numpy as np
import scipy.io
from dataset_loader import load_audio_files
from standard_nmf import standard_nmf
from alpha_nmf import alpha_nmf
from lingo_nmf import lingonmf
from evaluation import evaluate_separation
from nmf_utils import plot_results
from config import options

# Load dataset
X, fs = load_audio_files()

# Assume a predefined mixing matrix A
A = np.array([[1, 2], [2, 1]])

# Generate mixed signals
Y = 5 * np.dot(A, X) + 6

# Apply different NMF techniques

H_std, XH_std = standard_nmf(Y, options)
# Plot results
print("Standard NMF")
plot_results(Y, XH_std, X)

# Print results
sdr, sir, sar=evaluate_separation(X, XH_std)
print("SIR:", sir)
print("SDR:", sdr)
print("SAR:", sar)

H_alpha, XH_alpha = alpha_nmf(Y, options)
# Plot results
print("Alpha NMF")
plot_results(Y, XH_alpha, X)

# Print results
sdr, sir, sar=evaluate_separation(X, XH_alpha)
print("SIR:", sir)
print("SDR:", sdr)
print("SAR:", sar)


# Load the data from .mat files
XH_mat = scipy.io.loadmat('XH_Mohawk.mat')
XH = XH_mat['XH']
# Plot results
print("PL NMF")
plot_results(Y, XH, X)

# Print results
sdr, sir, sar=evaluate_separation(X, XH)
print("SIR:", sir)
print("SDR:", sdr)
print("SAR:", sar)

W_lingo, H_lingo, f0_penalty = lingonmf(Y, options, fs)
# Plot results
print("Lingo NMF")
plot_results(Y, H_lingo, X)

# Print results
sdr, sir, sar=evaluate_separation(X, H_lingo)
print("SIR:", sir)
print("SDR:", sdr)
print("SAR:", sar)
