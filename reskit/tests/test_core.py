import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import pytest
import numpy as np
import pandas as pd

from reskit.core import DataTransformer
from reskit.core import Pipeliner
from reskit.core import walker_by_zero_dim
from reskit.core import walker_by_ids
from reskit.normalizations import mean_norm
from reskit.normalizations import binar_norm
from reskit.features import bag_of_edges
from reskit.features import degrees

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_mean_std_params_for_best_clf(grid_clf):
    for i, params in enumerate(grid_clf.cv_results_['params']):

        if params == grid_clf.best_params_:
            best_params = {}
            for key in grid_clf.best_params_:
                best_params[key.split('__')[1]] = grid_clf.best_params_[key]
            return  grid_clf.cv_results_['mean_test_score'][i], \
                grid_clf.cv_results_['std_test_score'][i], \
                str(best_params)


def are_dicts_equal(dict_1, dict_2):

    if dict_1.keys() != dict_2.keys():
        return False

    equal = True
    for key in dict_1:
        if (dict_1[key] != dict_2[key]).all():
            return False

    return True


def are_matrices_equal(matrix_1, matrix_2):

    return (matrix_1 == matrix_2).all()


def test_DataTransformer_collect():

    matrix_0 = np.random.rand(5, 5)
    matrix_1 = np.random.rand(5, 5)
    matrix_2 = np.random.rand(5, 5)
    y = np.array([0, 0, 1])

    X = {'matrices': {'id1': matrix_0,
                      'id2': matrix_1,
                      'id3': matrix_2}}

    degrees_features = np.array(
        [degrees(X['matrices']['id1']),
         degrees(X['matrices']['id2']),
         degrees(X['matrices']['id3'])])

    bag_of_edges_features = np.array(
        [bag_of_edges(X['matrices']['id1']),
         bag_of_edges(X['matrices']['id2']),
         bag_of_edges(X['matrices']['id3'])])

    output_X = np.hstack((degrees_features, bag_of_edges_features))

    temp_X = DataTransformer(
        global_func=walker_by_ids,
        global__from_field='matrices',
        global__to_field='degrees',
        local_func=degrees).fit_transform(X)

    result_X = DataTransformer(
        global_func=walker_by_ids,
        global__from_field='matrices',
        global__to_field='bag_of_edges',
        global__collect=['degrees', 'bag_of_edges'],
        local_func=bag_of_edges).fit_transform(temp_X)

    assert are_matrices_equal(result_X, output_X)


def test_DataTransformer_simple_transformation():

    matrix_0 = np.random.rand(5, 5)
    matrix_1 = np.random.rand(5, 5)
    matrix_2 = np.random.rand(5, 5)
    y = np.array([0, 0, 1])

    X = {'matrices': {'id1': matrix_0,
                      'id2': matrix_1,
                      'id3': matrix_2}}

    output = {'matrices': {'id1': mean_norm(matrix_0),
                           'id2': mean_norm(matrix_1),
                           'id3': mean_norm(matrix_2)}}

    result = DataTransformer(
        global_func=walker_by_ids,
        global__from_field='matrices',
        global__to_field='matrices',
        local_func=mean_norm).fit_transform(X)

    assert are_dicts_equal(output['matrices'], result['matrices'])


def test_DataTransformer_walker_by_zero_dim():

    matrix_0 = np.random.rand(5, 5)
    matrix_1 = np.random.rand(5, 5)
    matrix_2 = np.random.rand(5, 5)

    X = np.array([matrix_0,
                  matrix_1,
                  matrix_2])
    y = np.array([0, 0, 1])

    output = np.array([mean_norm(matrix_0),
                       mean_norm(matrix_1),
                       mean_norm(matrix_2)])

    result = DataTransformer(
        global_func=walker_by_zero_dim,
        local_func=mean_norm).fit_transform(X)

    assert (output == result).all()


def test_Pipeliner_table_generation():

    data = dict(step1=['one', 'one', 'one', 'one', 'one', 'one',
                       'two', 'two', 'two', 'two', 'two', 'two'],
                step2=['one', 'one', 'one', 'two', 'two', 'two',
                       'one', 'one', 'one', 'two', 'two', 'two'],
                step3=['one', 'two', 'three', 'one', 'two', 'three',
                       'one', 'two', 'three', 'one', 'two', 'three'])

    output = pd.DataFrame(data=data)

    step1 = [('one', 0),
             ('two', 0)]

    step2 = [('one', 0),
             ('two', 0)]

    step3 = [('one', 0),
             ('two', 0),
             ('three', 0)]

    steps = [('step1', step1),
             ('step2', step2),
             ('step3', step3)]

    result = Pipeliner(steps=steps).plan_table

    assert ((output == result).all()).all()


def test_Pipeliner_simple_experiment():

    X, y = make_classification()

    pipeline0 = Pipeline([('Scaler', MinMaxScaler()),
                          ('Classifier', LogisticRegression())])

    pipeline1 = Pipeline([('Scaler', MinMaxScaler()),
                          ('Classifier', SVC())])

    pipeline2 = Pipeline([('Scaler', StandardScaler()),
                          ('Classifier', LogisticRegression())])

    pipeline3 = Pipeline([('Scaler', StandardScaler()),
                          ('Classifier', SVC())])

    param_grid_LR = {'Classifier__penalty': ['l1',
                                             'l2']}
    param_grid_SVC = {'Classifier__kernel': ['linear',
                                             'poly',
                                             'rbf',
                                             'sigmoid']}

    grid_clf0 = GridSearchCV(
        estimator=pipeline0,
        param_grid=param_grid_LR,
        n_jobs=-1)
    grid_clf1 = GridSearchCV(
        estimator=pipeline1,
        param_grid=param_grid_SVC,
        n_jobs=-1)
    grid_clf2 = GridSearchCV(
        estimator=pipeline2,
        param_grid=param_grid_LR,
        n_jobs=-1)
    grid_clf3 = GridSearchCV(
        estimator=pipeline3,
        param_grid=param_grid_SVC,
        n_jobs=-1)

    grid_clf0.fit(X, y)
    grid_clf1.fit(X, y)
    grid_clf2.fit(X, y)
    grid_clf3.fit(X, y)

    scores0 = cross_val_score(grid_clf0.best_estimator_, X, y, n_jobs=-1)
    scores1 = cross_val_score(grid_clf1.best_estimator_, X, y, n_jobs=-1)
    scores2 = cross_val_score(grid_clf2.best_estimator_, X, y, n_jobs=-1)
    scores3 = cross_val_score(grid_clf3.best_estimator_, X, y, n_jobs=-1)

    data = dict(Scaler=['minmax', 'minmax',
                        'standard', 'standard'],
                Classifier=['LR', 'SVC',
                            'LR', 'SVC'])

    scalers = [('minmax', MinMaxScaler()),
               ('standard', StandardScaler())]

    classifiers = [('LR', LogisticRegression()),
                   ('SVC', SVC())]

    steps = [('Scaler', scalers),
             ('Classifier', classifiers)]

    param_grid = {'LR': {'penalty': ['l1',
                                     'l2']},

                  'SVC': {'kernel': ['linear',
                                     'poly',
                                     'rbf',
                                     'sigmoid']}}

    output_grid0 = get_mean_std_params_for_best_clf(grid_clf0)
    output_grid1 = get_mean_std_params_for_best_clf(grid_clf1)
    output_grid2 = get_mean_std_params_for_best_clf(grid_clf2)
    output_grid3 = get_mean_std_params_for_best_clf(grid_clf3)

    pipe = Pipeliner(steps=steps,
                     param_grid=param_grid)

    result = pipe.get_results(X, y)

    result_grid0 = (result.grid_accuracy_mean.loc[0],
                    result.grid_accuracy_std.loc[0],
                    result.grid_accuracy_best_params.loc[0])
    result_grid1 = (result.grid_accuracy_mean.loc[1],
                    result.grid_accuracy_std.loc[1],
                    result.grid_accuracy_best_params.loc[1])
    result_grid2 = (result.grid_accuracy_mean.loc[2],
                    result.grid_accuracy_std.loc[2],
                    result.grid_accuracy_best_params.loc[2])
    result_grid3 = (result.grid_accuracy_mean.loc[3],
                    result.grid_accuracy_std.loc[3],
                    result.grid_accuracy_best_params.loc[3])

    assert all([output_grid0 == result_grid0])


def test_Pipeliner_forbidden_combinations():

    data = dict(step1=['one', 'one', 'two', 'two'],
                step2=['one', 'one', 'two', 'two'],
                step3=['one', 'three', 'two', 'three'])

    output = pd.DataFrame(data=data)

    step1 = [('one', 0),
             ('two', 0)]

    step2 = [('one', 0),
             ('two', 0)]

    step3 = [('one', 0),
             ('two', 0),
             ('three', 0)]

    steps = [('step1', step1),
             ('step2', step2),
             ('step3', step3)]

    banned_combos = [('one', 'two')]

    result = Pipeliner(steps=steps,
                       banned_combos=banned_combos).plan_table

    assert ((output == result).all()).all()


def test_Pipeliner_caching():

    X, y = make_classification()

    output = StandardScaler().fit_transform(X)

    scalers = [('standard', StandardScaler())]

    classifiers = [('LR', LogisticRegression())]

    steps = [('Scaler', scalers),
             ('Classifier', classifiers)]

    param_grid = {'LR': {'penalty': ['l1',
                                     'l2']}}

    pipe = Pipeliner(steps=steps,
                     param_grid=param_grid)

    pipe.get_results(X, y, caching_steps=['Scaler'])
    result = pipe._cached_X['standard']

    assert (output == result).all()
