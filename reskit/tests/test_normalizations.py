import pytest
import numpy as np
from reskit.normalizations import binar_norm
from reskit.normalizations import max_norm
from reskit.normalizations import mean_norm
from reskit.normalizations import spectral_norm
from reskit.normalizations import rwalk_norm
from reskit.normalizations import double_norm
from reskit.normalizations import sqrtw
from reskit.normalizations import invdist
from reskit.normalizations import rootwbydist
from reskit.normalizations import wbysqdist


def is_equal(matrix_1, matrix_2):

    return (matrix_1 == matrix_2).all()


def test_binar_norm():

    input_matrix_0 = np.array([[0.1, 0.0, 0.5],
                               [0.0, 0.5, 0.0],
                               [0.0, 0.0, 0.9]])

    output_matrix_0 = np.array([[1, 0, 1],
                                [0, 1, 0],
                                [0, 0, 1]])

    tests = [is_equal(binar_norm(input_matrix_0), output_matrix_0)]

    assert all(tests)


def test_max_norm():

    input_matrix_0 = np.array([[2, 2, 2],
                               [4, 4, 4],
                               [8, 8, 8]])

    output_matrix_0 = np.array([[0.25, 0.25, 0.25],
                                [0.50, 0.50, 0.50],
                                [1.00, 1.00, 1.00]])

    tests = [is_equal(max_norm(input_matrix_0), output_matrix_0)]

    assert all(tests)


def test_mean_norm():

    input_matrix_0 = np.array([[1, 1, 1],
                               [2, 2, 2],
                               [3, 3, 3]])

    output_matrix_0 = np.array([[0.5, 0.5, 0.5],
                                [1.0, 1.0, 1.0],
                                [1.5, 1.5, 1.5]])

    tests = [is_equal(mean_norm(input_matrix_0), output_matrix_0)]

    assert all(tests)
