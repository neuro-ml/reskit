import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import pytest
import numpy as np
import pandas as pd

from numpy import array
from scipy.stats import uniform

from reskit.core import DataTransformer
from reskit.core import MatrixTransformer
from reskit.core import Pipeliner
from reskit.core import NestedGridSearchCV
from reskit.normalizations import mean_norm
from reskit.normalizations import binar_norm
from reskit.features import bag_of_edges
from reskit.features import degrees

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold


grid_cv = StratifiedKFold(random_state=0)
eval_cv = StratifiedKFold(random_state=1)


def get_mean_std_params_for_best_clf(grid_clf):
    for i, params in enumerate(grid_clf.cv_results_['params']):

        if params == grid_clf.best_params_:
            best_params = {}
            for key in grid_clf.best_params_:
                best_params[key.split('__')[1]] = grid_clf.best_params_[key]
            return  grid_clf.cv_results_['mean_test_score'][i], \
                grid_clf.cv_results_['std_test_score'][i], \
                str(best_params)


def test_DataTransformer():

    matrix_0 = np.random.rand(5, 5)
    matrix_1 = np.random.rand(5, 5)
    matrix_2 = np.random.rand(5, 5)
    y = np.array([0, 0, 1])

    def transform_func(data):
        N = len(data['matrices'])
        for i in range(N):
            data['matrices'][i] = mean_norm(
                data['matrices'][i])
        return data

    X = {'matrices': np.array([matrix_0,
                               matrix_1,
                               matrix_2])}

    output = {'matrices': np.array([mean_norm(matrix_0),
                                    mean_norm(matrix_1),
                                    mean_norm(matrix_2)])}

    result = DataTransformer(
        func=transform_func).fit_transform(X)

    assert (output['matrices'] == result['matrices']).all()


def test_MatrixTransformer():

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

    result = MatrixTransformer(
        func=mean_norm).fit_transform(X)

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

    result = Pipeliner(steps=steps,
                       grid_cv=grid_cv,
                       eval_cv=eval_cv).plan_table

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
        n_jobs=-1,
        cv=grid_cv)
    grid_clf1 = GridSearchCV(
        estimator=pipeline1,
        param_grid=param_grid_SVC,
        n_jobs=-1,
        cv=grid_cv)
    grid_clf2 = GridSearchCV(
        estimator=pipeline2,
        param_grid=param_grid_LR,
        n_jobs=-1,
        cv=grid_cv)
    grid_clf3 = GridSearchCV(
        estimator=pipeline3,
        param_grid=param_grid_SVC,
        n_jobs=-1,
        cv=grid_cv)

    grid_clf0.fit(X, y)
    grid_clf1.fit(X, y)
    grid_clf2.fit(X, y)
    grid_clf3.fit(X, y)
    
    output_scores0 = cross_val_score(grid_clf0.best_estimator_, X, y, cv=eval_cv, n_jobs=-1)
    output_scores1 = cross_val_score(grid_clf1.best_estimator_, X, y, cv=eval_cv, n_jobs=-1)
    output_scores2 = cross_val_score(grid_clf2.best_estimator_, X, y, cv=eval_cv, n_jobs=-1)
    output_scores3 = cross_val_score(grid_clf3.best_estimator_, X, y, cv=eval_cv, n_jobs=-1)

    output_scores0 = eval(repr(output_scores0))
    output_scores1 = eval(repr(output_scores1))
    output_scores2 = eval(repr(output_scores2))
    output_scores3 = eval(repr(output_scores3))

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

    optimizer = GridSearchCV
    optimizer_param_dict = dict()

    optimizer_param_dict['LR'] = dict(n_jobs=-1,
                                      param_grid={'LR__penalty': ['l1',
                                                                  'l2']})

    optimizer_param_dict['SVC'] = dict(n_jobs=-1,
                                       param_grid={'SVC__kernel': ['linear',
                                                                   'poly',
                                                                   'rbf',
                                                                   'sigmoid']})

    output_grid0 = get_mean_std_params_for_best_clf(grid_clf0)
    output_grid1 = get_mean_std_params_for_best_clf(grid_clf1)
    output_grid2 = get_mean_std_params_for_best_clf(grid_clf2)
    output_grid3 = get_mean_std_params_for_best_clf(grid_clf3)

    pipe = Pipeliner(steps=steps,
                     grid_cv=grid_cv,
                     eval_cv=eval_cv,
                     optimizer=optimizer,
                     optimizer_param_dict=optimizer_param_dict)

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

    result_scores0 = eval(result.eval_accuracy_scores.loc[0])
    result_scores1 = eval(result.eval_accuracy_scores.loc[1])
    result_scores2 = eval(result.eval_accuracy_scores.loc[2])
    result_scores3 = eval(result.eval_accuracy_scores.loc[3])

    assert all([output_grid0 == result_grid0])
    assert all([output_grid1 == result_grid1])
    assert all([output_grid2 == result_grid2])
    assert all([output_grid3 == result_grid3])

    assert all(result_scores0 == output_scores0)
    assert all(result_scores1 == output_scores1)
    assert all(result_scores2 == output_scores2)
    assert all(result_scores3 == output_scores3)


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
                       grid_cv=grid_cv,
                       eval_cv=eval_cv,
                       banned_combos=banned_combos ).plan_table

    assert ((output == result).all()).all()


def test_Pipeliner_caching():

    X, y = make_classification()

    output = StandardScaler().fit_transform(X)

    scalers = [('standard', StandardScaler())]

    classifiers = [('LR', LogisticRegression())]

    steps = [('Scaler', scalers),
             ('Classifier', classifiers)]

    optimizer = GridSearchCV
    optimizer_param_dict = dict()

    optimizer_param_dict['LR'] = dict(n_jobs=-1,
                                      param_grid={'LR__penalty': ['l1',
                                                                  'l2']})

    pipe = Pipeliner(steps=steps,
                     grid_cv=grid_cv,
                     eval_cv=eval_cv,
                     optimizer=optimizer,
                     optimizer_param_dict=optimizer_param_dict)

    pipe.get_results(X, y, caching_steps=['Scaler'])
    result = pipe._cached_X['standard']

    assert (output == result).all()


def test_Pipeliner_eval_cv_None():
    
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
        n_jobs=-1,
        cv=grid_cv)
    grid_clf1 = GridSearchCV(
        estimator=pipeline1,
        param_grid=param_grid_SVC,
        n_jobs=-1,
        cv=grid_cv)
    grid_clf2 = GridSearchCV(
        estimator=pipeline2,
        param_grid=param_grid_LR,
        n_jobs=-1,
        cv=grid_cv)
    grid_clf3 = GridSearchCV(
        estimator=pipeline3,
        param_grid=param_grid_SVC,
        n_jobs=-1,
        cv=grid_cv)

    grid_clf0.fit(X, y)
    grid_clf1.fit(X, y)
    grid_clf2.fit(X, y)
    grid_clf3.fit(X, y)
    
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

    optimizer = GridSearchCV
    optimizer_param_dict = dict()

    optimizer_param_dict['LR'] = dict(n_jobs=-1,
                                      param_grid={'LR__penalty': ['l1',
                                                                  'l2']})

    optimizer_param_dict['SVC'] = dict(n_jobs=-1,
                                       param_grid={'SVC__kernel': ['linear',
                                                                   'poly',
                                                                   'rbf',
                                                                   'sigmoid']})

    output_grid0 = get_mean_std_params_for_best_clf(grid_clf0)
    output_grid1 = get_mean_std_params_for_best_clf(grid_clf1)
    output_grid2 = get_mean_std_params_for_best_clf(grid_clf2)
    output_grid3 = get_mean_std_params_for_best_clf(grid_clf3)
    
    pipe = Pipeliner(steps=steps,
                     grid_cv=grid_cv,
                     eval_cv=None,
                     optimizer=optimizer,
                     optimizer_param_dict=optimizer_param_dict)

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
    assert all([output_grid1 == result_grid1])
    assert all([output_grid2 == result_grid2])
    assert all([output_grid3 == result_grid3])

    assert 'eval_accuracy_mean' not in result.columns
    assert 'eval_accuracy_std' not in result.columns
    assert 'eval_accuracy_scores' not in result.columns


def test_Pipeliner_grid_cv_None():

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

    output_scores0 = cross_val_score(pipeline0, X, y, cv=eval_cv, n_jobs=-1)
    output_scores1 = cross_val_score(pipeline1, X, y, cv=eval_cv, n_jobs=-1)
    output_scores2 = cross_val_score(pipeline2, X, y, cv=eval_cv, n_jobs=-1)
    output_scores3 = cross_val_score(pipeline3, X, y, cv=eval_cv, n_jobs=-1)

    output_scores0 = eval(repr(output_scores0))
    output_scores1 = eval(repr(output_scores1))
    output_scores2 = eval(repr(output_scores2))
    output_scores3 = eval(repr(output_scores3))

    scalers = [('minmax', MinMaxScaler()),
               ('standard', StandardScaler())]

    classifiers = [('LR', LogisticRegression()),
                   ('SVC', SVC())]

    steps = [('Scaler', scalers),
             ('Classifier', classifiers)]

    optimizer = GridSearchCV

    pipe = Pipeliner(steps=steps,
                     grid_cv=None, 
                     eval_cv=eval_cv, 
                     optimizer=optimizer)

    result = pipe.get_results(X, y)

    result_scores0 = eval(result.eval_accuracy_scores.loc[0])
    result_scores1 = eval(result.eval_accuracy_scores.loc[1])
    result_scores2 = eval(result.eval_accuracy_scores.loc[2])
    result_scores3 = eval(result.eval_accuracy_scores.loc[3])

    assert all(result_scores0 == output_scores0)
    assert all(result_scores1 == output_scores1)
    assert all(result_scores2 == output_scores2)
    assert all(result_scores3 == output_scores3)
    assert 'grid_accuracy_mean' not in result.columns
    assert 'grid_accuracy_std' not in result.columns
    assert 'grid_accuracy_best_params' not in result.columns


def test_Pipeliner_optimizer_None():

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

    output_scores0 = cross_val_score(pipeline0, X, y, cv=eval_cv, n_jobs=-1)
    output_scores1 = cross_val_score(pipeline1, X, y, cv=eval_cv, n_jobs=-1)
    output_scores2 = cross_val_score(pipeline2, X, y, cv=eval_cv, n_jobs=-1)
    output_scores3 = cross_val_score(pipeline3, X, y, cv=eval_cv, n_jobs=-1)

    output_scores0 = eval(repr(output_scores0))
    output_scores1 = eval(repr(output_scores1))
    output_scores2 = eval(repr(output_scores2))
    output_scores3 = eval(repr(output_scores3))

    scalers = [('minmax', MinMaxScaler()),
               ('standard', StandardScaler())]

    classifiers = [('LR', LogisticRegression()),
                   ('SVC', SVC())]

    steps = [('Scaler', scalers),
             ('Classifier', classifiers)]

    optimizer = GridSearchCV

    pipe = Pipeliner(steps=steps,
                     grid_cv=grid_cv, 
                     eval_cv=eval_cv, 
                     optimizer=None)

    result = pipe.get_results(X, y)

    result_scores0 = eval(result.eval_accuracy_scores.loc[0])
    result_scores1 = eval(result.eval_accuracy_scores.loc[1])
    result_scores2 = eval(result.eval_accuracy_scores.loc[2])
    result_scores3 = eval(result.eval_accuracy_scores.loc[3])

    assert all(result_scores0 == output_scores0)
    assert all(result_scores1 == output_scores1)
    assert all(result_scores2 == output_scores2)
    assert all(result_scores3 == output_scores3)
    assert 'grid_accuracy_mean' not in result.columns
    assert 'grid_accuracy_std' not in result.columns
    assert 'grid_accuracy_best_params' not in result.columns

"""
def test_NedtedGridSearchCV():
    X, y = make_classification()
    nested_cv = StratifiedKFold(random_state=2)
    nested_grid = NestedGridSearchCV(cv=grid_cv,
                                     nested_cv=nested_cv,
                                     scoring='roc_auc',
                                     n_jobs=-1)
    nested_grid.fit(X, y)

    for train, test in cv.split(X, y):
        grid_clf = GridSearchCV(cv=nested_cv,
                                scoring='roc_auc',
                                n_jobs=-1)
        grid_clf.fit(X[train], y[train])
        scores.append(mean(grid

    assert nested_grid.best_params_ == output_best_params_
    assert nested_grid.best_estimators_ == output_best_estimators_
    assert nested_grid
"""
