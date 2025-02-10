# Copyright (c) 2025 Y. Torabi et al.
# Cite as: https://doi.org/10.48550/arXiv.2305.01889

# alpha_nmf.py
# This module performs Alpha-NMF with regularization on the H matrix.

import numpy as np
from sklearn.decomposition import NMF
import mir_eval

def perform_alpha_nmf(Y, options, X, fs):
    """Performs Alpha Non-Negative Matrix Factorization (NMF) on the mixed signals."""
    model = NMF(n_components=options['J'], init=options['init'], max_iter=options['niter'],
                alpha_H=options['alpha_H'])
    W = model.fit_transform(Y.T)
    H = model.components_
    
    # Reconstruct the signals from the components
    reconstructed_signals = np.dot(W, H)
    
    # Evaluate separation performance
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(X, reconstructed_signals)
    print("Alpha-NMF SIR:", sir)
    print("Alpha-NMF SDR:", sdr)
    print("Alpha-NMF SAR:", sar)
    
    return H, reconstructed_signals
