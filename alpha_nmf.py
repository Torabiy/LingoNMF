import numpy as np
from sklearn.decomposition import NMF

def alpha_nmf(Y, options):
    """
    Alpha NMF with specified regularization.
    """
    np.random.seed(22)
    model = NMF(n_components=options['J'], init=options['init'], max_iter=options['niter'], alpha_H=options['alpha_H'])
    W = model.fit_transform(Y.T)
    H = model.components_

    return H, W.T
