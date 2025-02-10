# pl_nmf.py
# Copyright (c) 2025 Y. Torabi et al.
# Cite as: https://doi.org/10.48550/arXiv.2305.01889

import numpy as np

# Function to initialize NMF matrices
def initialize_nmf(Y, rank):
    np.random.seed(42)
    I, T = Y.shape
    A = np.abs(np.random.randn(I, rank))  # Basis matrix
    X = np.abs(np.random.randn(rank, T))  # Coefficient matrix
    return A, X

# Function to perform multiplicative update rules
def update_rules(A, X, Y, num_steps=50):
    for _ in range(num_steps):
        AX = A @ X + 1e-10  # Avoid division by zero
        A *= (Y @ X.T) / (A @ (X @ X.T) + 1e-10)
        X *= (A.T @ Y) / ((A.T @ A) @ X + 1e-10)
    return A, X

# Multi-Layer NMF with Alpha-Divergence
def perform_pl_nmf(Y, options, X, fs):
    ranks = options.get('ranks', [4, 2])
    alpha = options.get('alpha', 0.5) 

    if not ranks:
        raise ValueError("Rank list cannot be empty")

    A_list, X_list = [], []
    X_prev = Y.copy()

    for rank in ranks:
        if X_prev.shape[1] < rank:
            raise ValueError(f"Rank {rank} is too large for input shape {X_prev.shape}")

        A, X = initialize_nmf(X_prev, rank)
        A, X = update_rules(A, X, X_prev, 50)  # 50 update steps
        A_list.append(A)
        X_list.append(X)
        X_prev = X  # Pass X to the next layer

    # Compute final A and X correctly
    A_final = A_list[0]
    for i in range(1, len(A_list)):
        A_final = A_final @ A_list[i]  # Multiply all A matrices in sequence

    X_final = X_list[-1]  # Last X is the final X

    return A_final, X_final

