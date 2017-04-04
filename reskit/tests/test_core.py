import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import pytest
import numpy as np
from reskit.core import DataTransformer
from reskit.normalizations import mean_norm
from reskit.features import bag_of_edges
from reskit.features import degrees


def is_dicts_equal(dict_1, dict_2):

    if dict_1.keys() != dict_2.keys():
        return False

    equal = True
    for key in dict_1:
        if (dict_1[key] != dict_2[key]).all():
            return False

    return True


def is_matrices_equal(matrix_1, matrix_2):

    return (matrix_1 == matrix_2).all()


def test_DataTransformer():

    matrix_0 = np.random.rand(5, 5)
    matrix_1 = np.random.rand(5, 5)
    matrix_2 = np.random.rand(5, 5)
    y = np.array([0, 0, 1])

    input_0 = {'matrices': {'id1': matrix_0,
                            'id2': matrix_1,
                            'id3': matrix_2},
               'y': y}

    output_0 = {'matrices': {'id1': mean_norm(matrix_0),
                             'id2': mean_norm(matrix_1),
                             'id3': mean_norm(matrix_2)},
                'y': y}

    degrees_features = np.array(
        [degrees(output_0['matrices']['id1']),
         degrees(output_0['matrices']['id2']),
         degrees(output_0['matrices']['id3'])])

    bag_of_edges_features = np.array(
        [bag_of_edges(output_0['matrices']['id1']),
         bag_of_edges(output_0['matrices']['id2']),
         bag_of_edges(output_0['matrices']['id3'])])

    output_1 = np.hstack((degrees_features, bag_of_edges_features))

    result_0 = DataTransformer(mean_norm).fit_transform(input_0)
    temp = DataTransformer(degrees, to_field='dgrs').fit_transform(result_0)
    result_1, y = DataTransformer(
        bag_of_edges,
        to_field='boe',
        collect=['dgrs', 'boe']).fit_transform(temp)

    tests = [is_dicts_equal(output_0['matrices'], result_0['matrices']),
             is_matrices_equal(result_1, output_1)]

    assert all(tests)
