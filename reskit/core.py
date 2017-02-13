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
from numpy import mean, std, hstack, zeros
from time import time

import os


class Pipeliner(object):
    """ 
    

    Parameters:
    -----------
    steps : list
        List of (name, transforms) tuples

    eval_cv : int, cross-validation generator or an iterable, optional
        Determines the evaluation cross-validation splitting strategy.
        Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - An object to be used as cross-validation generator.
            - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    grid_cv : int, cross-validation generator or an iterable, optional
         
    param_grid : 

    banned_combos : 
        
    Attributes:
    -----------

    Examples:
    ---------

    """
    def __init__(self, steps, eval_cv=None, grid_cv=None, param_grid=dict(),
            banned_combos=list()):
        steps = OrderedDict(steps)
        columns = list(steps)
        for column in columns:
            steps[column] = OrderedDict(steps[column])

        def accept_from_banned_combos(row_keys, banned_combo):
            if set(banned_combo) - set(row_keys) == set():
                return False
            else:
                return True

        column_keys = [list(steps[column]) for column in columns]
        plan_rows = list()
        for row_keys in product(*column_keys):
            accept = list()
            for bnnd_cmb in banned_combos:
                accept += [accept_from_banned_combos(row_keys, bnnd_cmb)]

            if all(accept):
                row_of_plan = OrderedDict()
                for column, row_key in zip(columns, row_keys):
                    row_of_plan[column] = row_key
                plan_rows.append(row_of_plan)

        self.plan_table = DataFrame().from_dict(plan_rows)[columns]
        self.named_steps = steps
        self.eval_cv = eval_cv
        self.grid_cv = grid_cv
        self.param_grid = param_grid
        self._cached_data = OrderedDict()
        self.best_params = dict()
        self.scores = dict()

    def dump(self, path_to_file):
        """
        Write a representation of a Pipeliner's class to a specified file.

        Parameters:
        -----------
        path_to_file : string
            A path to which you want to save a class.

        """
        with open(path_to_file, 'wb') as f:
            for attr in sorted(self.__dict__.keys()):
                dump(getattr(self, attr), f)

    def load(self, path_to_file):
        """
        Load a representation of a Pipeliner's class from a specified file.

        Parameters:
        -----------
        path_to_file : string
            A path from which you want to load a class.

        """
        with open(path_to_file, 'rb') as f:
            for attr in sorted(self.__dict__.keys()):
                setattr(self, attr, load(f))

    def transform_with_caching(self, data, row_keys):
        """
        Sequentially apply a list of transforms to an input data.

        Parameters:
        -----------
        data : any
            A data, to which you want to apply transformations.

        row_keys : list
            Keys of transformations.

        Returns:
        --------
        transformed_data : any
            A data transformed with all specified transformations.

        """
        columns = list(self.plan_table.columns[:len(row_keys)])

        def remove_unmatched_caching_data(row_keys):
            cached_keys = list(self._cached_data)
            unmatched_caching_keys = cached_keys.copy()
            for row_key, cached_key in zip(row_keys, cached_keys):
                if not row_key == cached_key:
                    break
                unmatched_caching_keys.remove(row_key)

            for unmatched_caching_key in unmatched_caching_keys:
                del self._cached_data[unmatched_caching_key]

        def transform_data_from_last_cached(row_keys, columns):
            prev_key = list(self._cached_data)[-1]
            for row_key, column in zip(row_keys, columns):
                transformer = self.named_steps[column][row_key]
                data = self._cached_data[prev_key]
                self._cached_data[row_key] = transformer.fit_transform(data)
                prev_key = row_key

        if 'init' not in self._cached_data:
            self._cached_data['init'] = data
            transform_data_from_last_cached(row_keys, columns)
        else:
            row_keys = ['init'] + row_keys
            columns = ['init'] + columns
            remove_unmatched_caching_data(row_keys)
            cached_keys = list(self._cached_data)
            cached_keys_length = len(cached_keys)
            for i in range(cached_keys_length):
                del row_keys[0]
                del columns[0]
            transform_data_from_last_cached(row_keys, columns)

        last_cached_key = list(self._cached_data)[-1]

        return self._cached_data[last_cached_key]

    def get_grid_search_results(self, X, y, row_keys, scoring):
        """
        Give results with tuned parameters of a specified pipeline using grid search. 

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and n_features is the number of features. 
            For kernel=”precomputed”, the expected shape of X is (n_samples, n_samples). 

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in regression)
        
        row_keys : list
            Keys of transformations.

        scoring : list or str
            Name of a scoring method.
            If None, the score method of the estimator is used.

        Returns:
        --------
        results : dict
            Results dictionary.
        """
        classifier_key = row_keys[-1]
        if classifier_key in self.param_grid:
            columns = list(self.plan_table.columns)[-len(row_keys):]

            steps = list()
            for row_key, column in zip(row_keys, columns):
                steps.append((row_key, self.named_steps[column][row_key]))

            param_grid = dict()
            for key, value in self.param_grid[classifier_key].items():
                param_grid['{}__{}'.format(classifier_key, key)] = value

            self.asdf = param_grid
            self.asdfasdf = self.param_grid[classifier_key]

            grid_clf = GridSearchCV(estimator=Pipeline(steps),
                                    param_grid=param_grid,
                                    scoring=scoring,
                                    n_jobs=-1,
                                    cv=self.grid_cv)
            grid_clf.fit(X, y)

            best_params = dict()
            classifier_key_len = len(classifier_key)
            for key, value in grid_clf.best_params_.items():
                key = key[classifier_key_len + 2:]
                best_params[key] = value
            param_key = ''.join(row_keys) + str(scoring)
            self.best_params[param_key] = best_params

            results = dict()
            for i, params in enumerate(grid_clf.cv_results_['params']):
                if params == grid_clf.best_params_:

                    k = 'grid_{}_mean'.format(scoring)
                    results[k] = grid_clf.cv_results_['mean_test_score'][i]

                    k = 'grid_{}_std'.format(scoring)
                    results[k] = grid_clf.cv_results_['std_test_score'][i]

                    k = 'grid_{}_best_params'.format(scoring)
                    results[k] = str(best_params)

            return results
        else:
            param_key = ''.join(row_keys) + str(scoring)
            self.best_params[param_key] = dict()
            results = dict()
            results['grid_{}_mean'.format(scoring)] = 'NaN'
            results['grid_{}_std'.format(scoring)] = 'NaN'
            results['grid_{}_best_params'.format(scoring)] = 'NaN'
            return results

    def get_scores(self, X, y, row_keys, scoring, collect_n=None):
        """
        

        Parameters:
        -----------

        X : 

        y : 

        row_keys : 

        scoring : 

        collect_n :


        Returns:
        --------

        """
        columns = list(self.plan_table.columns)[-len(row_keys):]
        param_key = ''.join(row_keys) + str(scoring)

        steps = list()
        for row_key, column in zip(row_keys, columns):
            steps.append((row_key, self.named_steps[column][row_key]))

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

    def get_results(self, data, caching_steps=list(), scoring='accuracy',
            results_file='results.csv', logs_file='results.log', collect_n=None):
        """ 


        Parameters:
        -----------
        data : 

        caching_steps : 

        scoring : 

        results_file : 

        logs_file : 
        
        collect_n :


        Returns:
        --------

        """
        if type(scoring) == str:
            scoring = [scoring]

        columns = list(self.plan_table.columns)
        without_caching = [step for step in columns
                                if step not in caching_steps]

        for metric in scoring:
            grid_steps = ['grid_{}_mean'.format(metric),
                          'grid_{}_std'.format(metric),
                          'grid_{}_best_params'.format(metric)]

            eval_steps = ['eval_{}_mean'.format(metric),
                          'eval_{}_std'.format(metric),
                          'eval_{}_scores'.format(metric)]

            columns += grid_steps + eval_steps

        results = DataFrame(columns=columns)

        try:
            os.remove(results_file)
            print('Removed previous results file -- {}.'.format(results_file))
        except:
            print('No previous results found.')
        if results_file != None:
            results.to_csv(results_file)

        columns = list(self.plan_table.columns)
        results[columns] = self.plan_table


        with open(logs_file, 'w+') as logs:
            N = len(self.plan_table.index)
            for idx in self.plan_table.index:
                print('Line: {}/{}'.format(idx + 1, N))
                logs.write('Line: {}/{}\n'.format(idx + 1, N))
                logs.write('{}\n'.format(str(self.plan_table.loc[idx])))
                row = self.plan_table.loc[idx]
                caching_keys = list(row[caching_steps].values)

                time_point = time()
                if caching_keys != []:
                    X_featured, y = self.transform_with_caching(data, caching_keys)
                else:
                    X_featured = data['X']
                    y = data['y']
                spent_time = round(time() - time_point, 3)
                logs.write('Got Features: {} sec\n'.format(spent_time))

                for metric in scoring:
                    logs.write('Scoring: {}\n'.format(metric))
                    ml_keys = list(row[without_caching].values)
                    time_point = time()
                    grid_res = self.get_grid_search_results(X_featured, y,
                                                            ml_keys,
                                                            metric)
                    spent_time = round(time() - time_point, 3)
                    logs.write('Grid Search: {} sec\n'.format(spent_time))
                    logs.write('Grid Search Results: {}\n'.format(grid_res))

                    for key, value in grid_res.items():
                        results.loc[idx][key] = value

                    time_point = time()
                    scores = self.get_scores(X_featured, y,
                                             ml_keys,
                                             metric,
                                             collect_n)
                    spent_time = round(time() - time_point, 3)
                    logs.write('Got Scores: {} sec\n'.format(spent_time))

                    mean_key = 'eval_{}_mean'.format(metric)
                    scores_mean = mean(scores)
                    results.loc[idx][mean_key] = scores_mean
                    logs.write('Scores mean: {}\n'.format(scores_mean))

                    std_key = 'eval_{}_std'.format(metric)
                    scores_std = std(scores)
                    results.loc[idx][std_key] = scores_std
                    logs.write('Scores std: {}\n'.format(scores_std))

                    scores_key = 'eval_{}_scores'.format(metric)
                    results.loc[idx][scores_key] = str(scores)
                    logs.write('Scores: {}\n\n'.format(str(scores)))
                results.loc[[idx]].to_csv(results_file, header=False, mode='a+')

        return results



class Transformer(TransformerMixin, BaseEstimator):
    """


    Parameters:
    -----------
    func : 

    params : 

    collect : 

    Attributes:
    -----------

    """
    def __init__(self, func, params=None, collect=None):
        self.func = func
        self.params = params
        self.collect = collect

    def fit(self, data):
        """
        

        Parameters:
        -----------
        data : 

        Returns:
        --------

        """
        return self

    def transform(self, data):
        """


        Parameters:
        -----------
        data : 

        Returns:
        --------
        """
        if type(data) == dict:
            data = data.copy()

        if self.params:
            self.params['data'] = data
            result = self.func(**self.params)
        else:
            result = self.func(data)

        if self.collect:
            y = result['y']
            X = result[self.collect[0]]

            for key in self.collect[1:]:
                X = hstack((X, result[key]))
            return X, y

        return result

__all__ = ['Transformer', 'Pipeliner']
