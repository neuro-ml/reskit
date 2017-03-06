=====================
Pipeliner Class Usage
=====================

.. code-block:: python

  from sklearn.datasets import make_classification

  X, y = make_classification()
  data = {'X': X, 'y': y}

1. Defining Pipelines Steps
---------------------------

.. code-block:: python

  import numpy as np
  from reskit.core import Transformer
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC


  def delete_first_column(data):
      data['X'] = np.delete(data['X'], 0, 1)
      return data

  def mean_scaling_for_last_column(data):
      mean_of_last_column = np.mean(data['X'][:,-1])
      data['X'][:,-1] = data['X'][:,-1] / mean_of_last_column
      return data

  transforms = [
      ('delete_first_column', delete_first_column),
      ('mean_scaling_for_last_column', mean_scaling_for_last_column)
  ]

  classifiers = [
          ('LR', LogisticRegression()),
          ('SVC', SVC())
  ]

  steps = [
          ('Transforms', transforms)
          ('Classifier', classifiers)
  ]

2. Defining Cross-Validations
-----------------------------

.. code-block:: python

  from sklearn.model_selection import StratifiedKFold


  grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

3. Defining Grid Search Parameters
----------------------------------

.. code-block:: python

  param_grid = {
          'LR' : {
                  'penalty' : ['l1', 'l2']
          },
          'SVC' : {
                  'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
          }
  }

4. Launching Experiment
-----------------------

.. code-block:: python

  pipe = Pipeliner(steps, eval_cv=eval_cv, grid_cv=grid_cv, param_grid=param_grid)
  pipe.get_results(data=data, scoring=['roc_auc'])
