import scipy
import numpy as np

def efficiency(X):
    A_inv = 1./X
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False, unweighted=False)
    inv_SPL_with_inf = 1./SPL
    inv_SPL_with_nan = inv_SPL_with_inf.copy()
    inv_SPL_with_nan[np.isinf(inv_SPL_with_inf)]=np.nan
    efs = np.nanmean(inv_SPL_with_nan, 1)
    return efs

__all__ = ['efficiency']
