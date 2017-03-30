import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import pytest
import numpy as np
from reskit.features import bag_of_edges
from reskit.features import closeness_centrality
from reskit.features import betweenness_centrality
from reskit.features import eigenvector_centrality
from reskit.features import pagerank
from reskit.features import degrees
from reskit.features import clustering_coefficient
from reskit.features import triangles
from reskit.features import efficiency

def is_equal(matrix_1, matrix_2):

    return (matrix_1 == matrix_2).all()


def test_bag_of_edges():
    input_matrix_0 = np.array([[0, 1, 2],
                               [0, 0, 3],
                               [0, 0, 0]])

    output_vector_0 = np.array([1, 2, 3])

    tests = [is_equal(bag_of_edges(input_matrix_0), output_vector_0)]

    assert all(tests)
