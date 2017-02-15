import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class l1_graph_kernel(BaseEstimator, TransformerMixin):
    """ 
    L1 graph kernel. 
    """
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        """
        Calculates the l2 graph kernel.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        y : array, shape (n_samples,)
            Class labels for samples in X.

        Returns
        -------
        kernel : {array-like}, shape (n_samples, n_samples)

        """
        N = len(X)
        kernel = np.zeros((N, N))

        for i in range(N):
            for j in range(i):
                kernel[i,j] = kernel[j,i] = np.abs(X[i] - X[j])

        return kernel
        
class l2_graph_kernel(BaseEstimator, TransformerMixin):
    """ 
    L1 graph kernel. 
    """
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        """
        Calculates the l1 graph kernel.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Samples.

        y : array, shape (n_samples,)
            Class labels for samples in X.

        Returns
        -------
        kernel : {array-like}, shape (n_samples, n_samples)
            
        """
        N = len(X)
        kernel = np.zeros((N, N))

        for i in range(N):
            for j in range(i):
                kernel[i,j] = kernel[j,i] = np.sqrt((X[i] - X[j])**2)

        return kernel
