""" Core classes. """


from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import check_scoring
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from collections import OrderedDict
from itertools import product
from pandas import DataFrame
from pickle import dump, load
from numpy import mean, std, hstack, vstack, zeros, array
from time import time

import os


def convert_steps_to_ordered_dict(steps):
    steps = OrderedDict(steps)
    columns = list(steps)
    for column in columns:
        steps[column] = OrderedDict(steps[column])
    return steps


def get_plan_table_column_keys(steps):
    columns = list(steps)
    column_keys = list()
    for column in columns:
        column_keys.append(list(steps[column]))
    return column_keys


def row_acceptance_for_bans(row_keys, banned_combos):
    for bnnd_cmb in banned_combos:
        if set(bnnd_cmb) - set(row_keys) == set():
            return False
    return True


def create_plan_table(steps, banned_combos):
    column_keys = get_plan_table_column_keys(steps)
    columns = list(steps)
    plan_rows = list()
    for row_keys in product(*column_keys):
        if row_acceptance_for_bans(row_keys, banned_combos):
            row_of_plan = OrderedDict()
            for column, row_key in zip(columns, row_keys):
                row_of_plan[column] = row_key
            plan_rows.append(row_of_plan)
    plan_table = DataFrame().from_dict(plan_rows)[columns]
    return plan_table


class Pipeliner(object):
    """
    An object which allows you to test different data preprocessing
    pipelines and prediction models at once.

    You will need to specify a name of each preprocessing and prediction
    step and possible objects performing each step. Then Pipeliner will
    combine these steps to different pipelines, excluding forbidden
    combinations; perform experiments according to these steps and present
    results in convenient csv table. For example, for each pipeline's
    classifier, Pipeliner will grid search on cross-validation to find the best
    classifier's parameters and report metric mean and std for each tested
    pipeline. Pipeliner also allows you to cache interim calculations to
    avoid unnecessary recalculations.

    Parameters
    ----------
    steps : list of tuples
        List of (step_name, transformers) tuples, where transformers is a
        list of tuples (step_transformer_name, transformer). ``Pipeliner``
        will create ``plan_table`` from this ``steps``, combining all
        possible combinations of transformers, switching transformers on
        each step.

    eval_cv : int, cross-validation generator or an iterable, optional
        Determines the evaluation cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - An object to be used as cross-validation generator.
            - A list or iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y``
        is either binary or multiclass, ``StratifiedKFold`` is used. In all
        other cases, ``KFold`` is used.

        Refer scikit-learn ``User Guide`` for the various cross-validation strategies that
        can be used here.

    grid_cv : int, cross-validation generator or an iterable, optional
         Determines the grid search cross-validation splitting strategy.
         Possible inputs for cv are the same as for ``eval_cv``.

    param_grid : dict of dictionaries
        Dictionary with classifiers names (string) as keys. The keys are
        possible classifiers names in ``steps``. Each key corresponds to
        grid search parameters.

    banned_combos : list of tuples
        List of (transformer_name_1, transformer_name_2) tuples. Each row
        with both transformers will be removed from ``plan_table``.

    Attributes
    ----------
    plan_table : pandas DataFrame
        Plan of pipelines evaluation. Created from ``steps``.

    named_steps: dict of dictionaries
        Dictionary with steps names as keys. Each key corresponds to
        dictionary with transformers names from ``steps`` as keys.
        You can get any transformer object from this dictionary.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.svm import SVC
    >>> from reskit.core import Pipeliner

    >>> X, y = make_classification()

    >>> scalers = [('minmax', MinMaxScaler()), ('standard', StandardScaler())]
    >>> classifiers = [('LR', LogisticRegression()), ('SVC', SVC())]
    >>> steps = [('Scaler', scalers), ('Classifier', classifiers)]

    >>> grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    >>> eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    >>> param_grid = {'LR' : {'penalty' : ['l1', 'l2']},
    >>>               'SVC' : {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}}

    >>> pipe = Pipeliner(steps, eval_cv=eval_cv, grid_cv=grid_cv, param_grid=param_grid)
    >>> pipe.get_results(X=X, y=y, scoring=['roc_auc'])
    """

    def __init__(
            self,
            steps,
            grid_cv,
            eval_cv,
            optimizer=None,
            optimizer_param_dict=dict(),
            banned_combos=list()):
        assert grid_cv is not None or eval_cv is not None, \
            'Enter at least grid_cv or eval_cv.'

        steps = convert_steps_to_ordered_dict(steps)

        self.plan_table = create_plan_table(steps, banned_combos)
        self.named_steps = steps
        self.eval_cv = eval_cv
        self.grid_cv = grid_cv
        self.optimizer = optimizer
        self.optimizer_param_dict = optimizer_param_dict
        self._cached_X = OrderedDict()
        self.best_params = dict()

    def _remove_unmatched_caching_X(self, row_keys):
        cached_keys = list(self._cached_X)
        unmatched_caching_keys = cached_keys.copy()
        for row_key, cached_key in zip(row_keys, cached_keys):
            if not row_key == cached_key:
                break
            unmatched_caching_keys.remove(row_key)

        for unmatched_caching_key in unmatched_caching_keys:
            del self._cached_X[unmatched_caching_key]

    def _transform_X_from_last_cached(self, row_keys, columns):
        prev_key = list(self._cached_X)[-1]
        for row_key, column in zip(row_keys, columns):
            transformer = self.named_steps[column][row_key]
            X = self._cached_X[prev_key]
            self._cached_X[row_key] = transformer.fit_transform(X)
            prev_key = row_key

    def _transform_with_caching(self, X, y, row_keys):
        columns = list(self.plan_table.columns[:len(row_keys)])
        if 'init' not in self._cached_X:
            self._cached_X['init'] = X
            self._transform_X_from_last_cached(row_keys, columns)
        else:
            row_keys = ['init'] + row_keys
            columns = ['init'] + columns
            self._remove_unmatched_caching_X(row_keys)
            cached_keys = list(self._cached_X)
            cached_keys_length = len(cached_keys)
            for i in range(cached_keys_length):
                del row_keys[0]
                del columns[0]
            self._transform_X_from_last_cached(row_keys, columns)
        last_cached_key = list(self._cached_X)[-1]
        return self._cached_X[last_cached_key], y

    def _create_pipeline_steps_for_grid_search(self, row_keys):
        columns = list(self.plan_table.columns)[-len(row_keys):]
        steps = list()
        for row_key, column in zip(row_keys, columns):
            steps.append((row_key, self.named_steps[column][row_key]))
        return steps

    def _get_param_key(self, row_keys, scoring):
        return ''.join(row_keys) + str(scoring)

    def _get_best_params(self, opt_obj, row_keys, scoring):
        classifier_key = row_keys[-1]
        best_params = dict()
        classifier_key_len = len(classifier_key)
        for key, value in opt_obj.best_params_.items():
            key = key[classifier_key_len + 2:]
            best_params[key] = value
        return best_params

    def _get_results_dict(self, opt_obj, param_key, scoring):
        results = dict()
        for i, params in enumerate(opt_obj.cv_results_['params']):
            if params == opt_obj.best_params_:

                k = 'grid_{}_mean'.format(scoring)
                results[k] = opt_obj.cv_results_['mean_test_score'][i]

                k = 'grid_{}_std'.format(scoring)
                results[k] = opt_obj.cv_results_['std_test_score'][i]

                k = 'grid_{}_best_params'.format(scoring)
                results[k] = str(self.best_params[param_key])
        return results

    def _get_grid_search_results(self, X, y, row_keys, scoring):

        classifier_key = row_keys[-1]
        if classifier_key in self.optimizer_param_dict:
            steps = self._create_pipeline_steps_for_grid_search(row_keys)

            opt_obj = self.optimizer(
                estimator=Pipeline(steps),
                scoring=scoring,
                cv=self.grid_cv,
                **self.optimizer_param_dict[classifier_key])
            opt_obj.fit(X, y)

            param_key = self._get_param_key(row_keys, scoring)
            self.best_params[param_key] = self._get_best_params(
                opt_obj, row_keys, scoring)
            results = self._get_results_dict(opt_obj, param_key, scoring)

            return results
        else:
            param_key = self._get_param_key(row_keys, scoring)
            self.best_params[param_key] = dict()
            results = dict()
            return results

    def _get_scores(self, X, y, row_keys, scoring, collect_n=None):
        param_key = self._get_param_key(row_keys, scoring)
        steps = self._create_pipeline_steps_for_grid_search(row_keys)

        steps[-1][1].set_params(**self.best_params[param_key])

        if not collect_n:
            scores = cross_val_score(Pipeline(steps), X, y,
                                     scoring=scoring,
                                     cv=self.eval_cv,
                                     n_jobs=-1)
        else:
            init_random_state = self.eval_cv.random_state
            scores = list()
            for i in range(collect_n):
                fold_prediction = cross_val_predict(Pipeline(steps), X, y,
                                                    cv=self.eval_cv,
                                                    n_jobs=-1)
                metric = check_scoring(steps[-1][1],
                                       scoring=scoring).__dict__['_score_func']
                scores.append(metric(y, fold_prediction))
                self.eval_cv.random_state += 1

            self.eval_cv.random_state = init_random_state
        return scores

    def _get_steps_without_caching(self, caching_steps):
        columns = list(self.plan_table.columns)
        without_caching = [step for step in columns
                           if step not in caching_steps]
        return without_caching

    def _create_results_dataframe(self, scoring):
        columns = list(self.plan_table.columns)
        for metric in scoring:
            if self.grid_cv is not None and \
               self.optimizer is not None:
                grid_steps = ['grid_{}_mean'.format(metric),
                              'grid_{}_std'.format(metric),
                              'grid_{}_best_params'.format(metric)]
                columns += grid_steps

            if self.eval_cv is not None:
                eval_steps = ['eval_{}_mean'.format(metric),
                              'eval_{}_std'.format(metric),
                              'eval_{}_scores'.format(metric)]
                columns += eval_steps

        results = DataFrame(columns=columns)
        columns = list(self.plan_table.columns)
        results[columns] = self.plan_table
        return results

    def _write_line_info(self, logs, idx):
        N = len(self.plan_table.index)
        print('Line: {}/{}'.format(idx + 1, N))
        logs.write('Line: {}/{}\n'.format(idx + 1, N))
        logs.write('{}\n'.format(str(self.plan_table.loc[idx])))

    def _get_caching_keys(self, idx, caching_steps):
        row = self.plan_table.loc[idx]
        caching_keys = list(row[caching_steps].values)
        return caching_keys

    def _write_spent_time(self, logs, time_point, message):
        spent_time = round(time() - time_point, 3)
        logs.write('{}: {} sec\n'.format(message, spent_time))

    def _get_ml_keys(self, idx, steps_without_caching):
        row = self.plan_table.loc[idx]
        ml_keys = list(row[steps_without_caching].values)
        return ml_keys

    def _get_model(self, classifier_key):
        column = list(self.plan_table.columns)[-1]
        return self.named_steps[columns][classifier_key]

    def get_results(self, X, y=None, caching_steps=list(), scoring='accuracy',
                    logs_file='results.log', collect_n=None):
        """
        Gives results dataframe by defined pipelines.

        Parameters
        ----------

        X : array-like
            The data to fit. Can be, for example a list, or an array at least 2d, or
            dictionary.

        y : array-like, optional, default: None
            The target variable to try to predict in the case of supervised learning.

        caching_steps : list of strings
            Steps which won’t be recalculated for each new pipeline.
            If in previous pipeline exists the same steps, ``Pipeliner``
            will start from this step.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer
            callable object / function with signature
            ``scorer(estimator, X, y)``. If None, the score method of
            the estimator is used.

        logs_file : string
            File name where logs will be saved.

        collect_n : int
            If not None scores will be calculated in following way. Each
            score will be corresponds to average score on cross-validation
            scores. The only thing that is changing for each score is
            random_state, it shifts.

        Returns
        -------
        results : DataFrame
            Dataframe with all results about pipelines.
        """

        if isinstance(scoring, str):
            scoring = [scoring]

        steps_without_caching = self._get_steps_without_caching(caching_steps)
        results = self._create_results_dataframe(scoring)

        with open(logs_file, 'w+') as logs:
            for idx in self.plan_table.index:

                self._write_line_info(logs, idx)
                caching_keys = self._get_caching_keys(idx, caching_steps)

                time_point = time()
                X_featured, y = self._transform_with_caching(
                    X, y, caching_keys)
                self._write_spent_time(logs, time_point, 'Got Features')

                for metric in scoring:
                    logs.write('Scoring: {}\n'.format(metric))

                    ml_keys = self._get_ml_keys(idx, steps_without_caching)

                    if self.grid_cv is not None and \
                       self.optimizer is not None:
                        time_point = time()
                        grid_res = self._get_grid_search_results(X_featured, y,
                                                                 ml_keys,
                                                                 metric)
                        self._write_spent_time(logs, time_point, 'Grid Search')
                        logs.write(
                            'Grid Search Results: {}\n'.format(grid_res))

                        for key, value in grid_res.items():
                            results.loc[idx][key] = value
                    else:
                        param_key = self._get_param_key(ml_keys, metric)
                        self.best_params[param_key] = dict()

                    mean_key = 'eval_{}_mean'.format(metric)
                    std_key = 'eval_{}_std'.format(metric)
                    scores_key = 'eval_{}_scores'.format(metric)

                    if self.eval_cv is not None:
                        time_point = time()
                        scores = self._get_scores(X_featured, y,
                                                  ml_keys,
                                                  metric,
                                                  collect_n)
                        self._write_spent_time(logs, time_point, 'Got Scores')

                        scores_mean = mean(scores)
                        scores_std = std(scores)

                        results.loc[idx][mean_key] = scores_mean
                        logs.write('Scores mean: {}\n'.format(scores_mean))

                        results.loc[idx][std_key] = scores_std
                        logs.write('Scores std: {}\n'.format(scores_std))

                        results.loc[idx][scores_key] = repr(scores)
                        logs.write('Scores: {}\n\n'.format(repr(scores)))

        return results


class MatrixTransformer(TransformerMixin, BaseEstimator):
    """
    Helps to add you own transformation through usual functions.

    Parameters
    ----------

    func : function
        A function that transforms input data.

    params : dict
        Parameters for the function.
    """

    def __init__(self, func, **params):
        self.func = func
        self.params = params

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        new_X = []
        for i in range(len(X)):
            new_X.append(self.func(X[i], **self.params))
        return array(new_X)


class DataTransformer(TransformerMixin, BaseEstimator):
    """
    Helps to add you own transformation through usual functions.

    Parameters
    ----------

    func : function
        A function that transforms input data.

    params : dict
        Parameters for the function.
    """

    def __init__(self, func, **params):
        self.func = func
        self.params = params

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        return self.func(X, **self.params)


class NestedGridSearchCV(BaseEstimator):

    def __init__(self, cv, nested_cv, **grid_param_dict):
        self.cv = cv
        self.nested_cv = nested_cv
        self.grid_param_dict = grid_param_dict
        self.best_estimators_ = list()
        self.best_params_ = list()
        self.cv_results_ = dict()

    def fit(self, X, y=None, **fit_params):
        for train, test in cv.split(X, y):
            grid_obj = GridSearchCV(cv=nested_cv,
                                    param_grid=self.grid_param_dict)
            grid_obj.fit(X[train], y[train])
            self.best_params_.append(grid_obj.best_params_)
        return self

    def predict(self, X, y=None):
        pass


__all__ = ['MatrixTransformer',
           'DataTransformer',
           'Pipeliner']
