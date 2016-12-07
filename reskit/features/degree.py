import numpy as np

def degrees(data):
    if data['X'].ndim == 3:
        data['degrees'] = np.sum(data['X'], axis=1)
    elif data['X'].ndim == 2:
        data['degrees'] = np.sum(data['X'], axis=1)
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size). ')

    return data

__all__ = ['degrees']
