""" Functions of norms. """

import numpy as np


def binar_norm(data):
    """
    Binary norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _binar_norm(X):
        bin_X = X.copy()
        bin_X[bin_X > 0] = 1
        return bin_X

    if data['X'].ndim == 2:
        data['X'] = _binar_norm(data['X'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_binar_norm(x)
                              for x in data['X']])
    return data


def max_norm(data):
    """
    Max norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _max_norm(X):
        normed_X = X / np.max(X)
        return normed_X

    if data['X'].ndim == 2:
        data['X'] = _max_norm(data['X'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_max_norm(x)
                              for x in data['X']])
    return data


def mean_norm(data):
    """
    Mean norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _mean_norm(X):
        normed_X = X / np.mean(X)
        return normed_X

    if data['X'].ndim == 2:
        data['X'] = _mean_norm(data['X'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_mean_norm(x)
                              for x in data['X']])
    return data


def spectral_norm(data):
    """
    Spectral norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _spectral_norm(X):
        np.fill_diagonal(X, 0)
        degrees = np.diag(1 / np.sqrt(np.nansum(X, axis=1)))
        normed_X = degrees.dot(X).dot(degrees)
        return normed_X

    if data['X'].ndim == 2:
        data['X'] = _spectral_norm(data['X'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_spectral_norm(x)
                              for x in data['X']])
    return data


def rwalk_norm(data):
    """
    Rwalk norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _rwalk_norm(X):
        degrees = np.diag(np.sum(X, axis=1))
        degrees = np.linalg.inv(degrees)
        normed_X = degrees.dot(X)
        return normed_X

    if data['X'].ndim == 2:
        data['X'] = _rwalk_norm(data['X'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_rwalk_norm(x)
                              for x in data['X']])
    return data


def double_norm(function, X1, X2):
    """
    Double norm.

    Parameters
    ----------
    function : function
    X1 : 1-st function input
    X2 : 2-nd function input

    Returns
    -------
    normed_X1, normed_X2 : 1-st function output, 2-nd function output
    """
    return function(X1), function(X2)


def sqrtw(data):
    """
    Square root norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _sqrtw(X):
        weighted_X = np.sqrt(X)
        return weighted_X

    if data['X'].ndim == 2:
        data['X'] = _sqrtw(data['X'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_sqrtw(x)
                              for x in data['X']])
    return data


def invdist(data):
    """
    Inverse distance norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _invdist(X, dist):
        bin_X = X.copy()
        bin_X[bin_X > 0] = 1
        weighted_X = bin_X / dist
        np.fill_diagonal(weighted_X, 0)
        return weighted_X

    if data['X'].ndim == 2:
        data['X'] = _invdist(data['X'], data['dist'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_invdist(x, dist)
                              for x, dist in zip(data['X'], data['dist'])])
    return data


def rootwbydist(data):
    """
    Root weight by distance norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _rootwbydist(X, dist):
        np.fill_diagonal(dist, 1)
        weighted_X = np.sqrt(X) / dist
        return weighted_X

    if data['X'].ndim == 2:
        data['X'] = _rootwbydist(data['X'], data['dist'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_rootwbydist(x, dist)
                              for x, dist in zip(data['X'], data['dist'])])
    return data


def wbysqdist(data):
    """
    Weights by squared distance norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _wbysqdist(X, dist):
        weighted_X = X / dist ** 2
        np.fill_diagonal(weighted_X, 0)
        return weighted_X

    if data['X'].ndim == 2:
        data['X'] = _wbysqdist(data['X'], data['dist'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_wbysqdist(x, dist)
                              for x, dist in zip(data['X'], data['dist'])])
    return data


def neg_log(data):
    """
    Negative logarithm norm.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    normed_data : dict
        Dictionary with normed matrix on field ``X``.
    """
    def _neg_log(X):
        normed_X = -np.lob(X)
        return normed_X

    if data['X'].ndim == 2:
        data['X'] = _neg_log(data['X'])
    elif data['X'].ndim == 3:
        data['X'] = np.array([_neg_log(x)
                              for x in data['X']])
    return data


__all__ = ['binar_norm',
           'max_norm',
           'mean_norm',
           'spectral_norm',
           'rwalk_norm',
           'double_norm',
           'sqrtw',
           'invdist',
           'rootwbydist',
           'wbysqdist']
