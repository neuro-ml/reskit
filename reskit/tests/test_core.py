import pytest
import numpy as np
from reskit.core import DataTransformer
from reskit.normalizations import mean_norm


def is_dicts_equal(dict_1, dict_2):
    equal = True
    for key in dict_1:
        if dict_1[key] != dict_2[key]:
            return False
    return True


def test_DataTransformer():
    matrix_0 = np.random.rand(5, 5)
    matrix_1 = np.random.rand(5, 5)
    matrix_2 = np.random.rand(5, 5)

    input_0 = {'matrices': {'id1': matrix_0,
                            'id2': matrix_1,
                            'id3': matrix_2}}

    output_0 = {'matrices': {'id1': mean_norm(matrix_0),
                             'id2': mean_norm(matrix_1),
                             'id3': mean_norm(matrix_2)}}['matrices']

    result_0 = DataTransformer(mean_norm).fit_transform(input_0)['matrices']

    tests = [result_0.keys() == output_0.keys()]

    assert all(tests)
