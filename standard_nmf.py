import numpy as np
from sklearn.decomposition import NMF

def standard_nmf(Y, options):
    """
    Standard Non-Negative Matrix Factorization (NMF).
    """
    np.random.seed(45)
    model = NMF(n_components=options['J'], init='random', max_iter=options['niter'])
    W = model.fit_transform(Y.T)
    H = model.components_

    return H, W.T
