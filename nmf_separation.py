# Copyright (c) 2025 Y. Torabi et al.
# Cite as: https://doi.org/10.48550/arXiv.2305.01889

# nmf_separation.py
# This module performs NMF and evaluates separation quality.

import numpy as np
from sklearn.decomposition import NMF
import mir_eval

def perform_nmf(Y, options, X, fs):
    """Performs Non-Negative Matrix Factorization (NMF) on the mixed signals."""
    model = NMF(n_components=options['J'], init='random', max_iter=options['niter'])
    W = model.fit_transform(Y.T)
    H = model.components_
    
    # Reconstruct the signals from the components
    reconstructed_signals = np.dot(W, H)
    
    # Evaluate separation performance
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(X, reconstructed_signals)
    print("SIR:", sir)
    print("SDR:", sdr)
    print("SAR:", sar)
    
    return H, reconstructed_signals
