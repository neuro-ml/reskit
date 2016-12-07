import numpy as np
import pandas as pd

def bag_of_edges(X, SPL=None, symmetric = True, return_df = False, offset = 1):
    size = X.shape[1]
    if symmetric:
        indices = np.triu_indices(size, k = offset)
    else:
        grid = np.indices(X.shape[1:])
        indices = (grid[0].reshape(-1), grid[1].reshape(-1))
    if len(X.shape) == 3:
        featurized_X = X[:, indices[0], indices[1]]
    elif len(X.shape) == 2:
        featurized_X = X[indices[0], indices[1]]
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size).')
    if return_df:
        col_names = ['edge_' + str(i) + '_' + str(j) for i,j in zip(indices[0], indices[1])]
        featurized_X = pd.DataFrame(featurized_X, columns=col_names)
    return featurized_X

__all__ = ['bag_of_edges']
