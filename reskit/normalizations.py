""" Functions of norms. """

import numpy as np


def binar_norm(X):
    """
    Binary matrix normalization.

    Transforms a matrix to binarized one.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to binarize.

    Returns
    -------
    normed_X : numpy matrix
        Binarized matrix.
    """
    bin_X = X.copy()
    bin_X[bin_X > 0] = 1
    return bin_X


def max_norm(X):
    """
    Maximum matrix normalization.

    Transforms a matrix to normalized by the maximum 
    matrix value.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    normed_X = X / np.max(X)
    return normed_X


def mean_norm(X):
    """
    Mean matrix normalization.

    Transforms a matrix to normalized by the mean
    matrix value.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    normed_X = X / np.mean(X)
    return normed_X


def spectral_norm(X):
    """
    Spectral matrix normalization.

    Transforms a matrix to normalized by the geometric mean
    of adjacent degrees.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    np.fill_diagonal(X, 0)
    degrees = np.diag(1 / np.sqrt(np.nansum(X, axis=1)))
    normed_X = degrees.dot(X).dot(degrees)
    return normed_X


def rwalk_norm(X):
    """
    Random walk matrix normalization.

    Transforms a matrix to normalized by degree of
    a destination node.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    degrees = np.diag(np.sum(X, axis=1))
    degrees = np.linalg.inv(degrees)
    normed_X = degrees.dot(X)
    return normed_X


def double_norm(function, X1, X2):
    """
    Double normalization.

    Applies normalization function to two matrices.

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


def sqrtw(X):
    """
    Square root matrix normalization.

    Transforms each matrix value to square root of this value.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    weighted_X = np.sqrt(X)
    return weighted_X


def invdist(X, dist):
    """
    Inverse distance matrix normalization.

    Inverces each matrix number.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    bin_X = X.copy()
    bin_X[bin_X > 0] = 1
    weighted_X = bin_X / dist
    np.fill_diagonal(weighted_X, 0)
    return weighted_X


def rootwbydist(X, dist):
    """
    Root weight by distance matrix normalization.

    Transforms each matrix value to square root of this value
    normalized by some defined weight.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    np.fill_diagonal(dist, 1)
    weighted_X = np.sqrt(X) / dist
    return weighted_X


def wbysqdist(X, dist):
    """
    Weights by squared distance matrix normalization.

    Transforms a matrix to normalized by square root of
    some defined weight.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    weighted_X = X / dist ** 2
    np.fill_diagonal(weighted_X, 0)
    return weighted_X


def neg_log(X):
    """
    Negative logarithm matrix normalization.

    Transforms each matrix value to negative logarithm
    of this value.

    Parameters
    ----------
    X : numpy matrix
        Matrix you want to normalize.

    Returns
    -------
    normed_X : numpy matrix
        Normalized matrix.
    """
    normed_X = -np.lob(X)
    return normed_X


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
