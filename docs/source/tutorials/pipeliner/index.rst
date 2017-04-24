=====================
Pipeliner Class Usage
=====================

The task is simple: find the best combination of pre-processing steps and
predictive models with respect to an objective criterion. Logistically this can
be problematic: a small example might involve three classification models, and
two data preprocessing steps with two possible variations for each — overall 12
combinations. For each of these combinations we would like to perform a grid
search of predefined hyperparameters on a fixed cross-validation dataset,
computing performance metrics for each option (for example ROC AUC). Clearly
this can become complicated quickly. On the other hand, many of these
combinations share substeps, and re-running such shared steps amounts to a loss
of compute time.

1. Defining Pipelines Steps and Grid Search Parameters
------------------------------------------------------

The researcher specifies the possible processing steps and the scikit objects
involved, then Reskit expands these steps to each possible pipeline. Reskit
represents these pipelines in a convenient pandas dataframe, so the researcher
can directly visualize and manipulate the experiments.

.. code-block:: python

  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import MinMaxScaler

  from sklearn.linear_model import LogisticRegression
  from sklearn.linear_model import SGDClassifier
  from sklearn.svm import SVC

  from sklearn.feature_selection import VarianceThreshold
  from sklearn.decomposition import PCA

  from sklearn.model_selection import StratifiedKFold

  from reskit.core import Pipeliner

  # Feature selection and feature extraction step variants (1st step)
  feature_engineering = [('VT', VarianceThreshold()),
                         ('PCA', PCA())]

  # Preprocessing step variants (2nd step)
  scalers = [('standard', StandardScaler()),
             ('minmax', MinMaxScaler())]

  # Models (3rd step)
  classifiers = [('LR', LogisticRegression()),
                 ('SVC', SVC()),
                 ('SGD', SGDClassifier())]

  # Reskit needs to define steps in this manner
  steps = [('feature_engineering', feature_engineering),
           ('scaler', scalers),
           ('classifier', classifiers)]

  # Grid search parameters for our models
  param_grid = {'LR': {'penalty': ['l1', 'l2']},
                'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                'SGD': {'penalty': ['elasticnet'],
                        'l1_ratio': [0.1, 0.2, 0.3]}}

  # Quality metric that we want to optimize
  scoring='roc_auc'

  # Setting cross-validations
  grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

  pipe = Pipeliner(steps=steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
  pipe.plan_table


.. csv-table::
  :file: pipeliner_defining.csv


2. Forbidden combinations
-------------------------

In case you don't want to use minmax scaler with SVC, you can define banned
combo:

.. code-block:: python
  banned_combos = [('minmax', 'SVC')]
  pipe = Pipeliner(steps=steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid, banned_combos=banned_combos)
  pipe.plan_table


.. csv-table::
  :file: pipeliner_forbidden.csv


3. Launching Experiment
-----------------------

Reskit then runs each experiment and presents results which are provided to the
user through a pandas dataframe. For each pipeline’s classifier, Reskit grid
search on cross-validation to find the best classifier’s parameters and report
metric mean and standard deviation for each tested pipeline (ROC AUC in this
case).

.. code-block:: python

  from sklearn.datasets import make_classification


  X, y = make_classification()
  pipe.get_results(X, y, scoring=['roc_auc'])

.. code-block:: bash

  Line: 1/10
  Line: 2/10
  Line: 3/10
  Line: 4/10
  Line: 5/10
  Line: 6/10
  Line: 7/10
  Line: 8/10
  Line: 9/10
  Line: 10/10


.. csv-table::
  :file: pipeliner_launching.csv


4. Caching intermediate steps
-----------------------------

Reskit also allows you to cache interim calculations to avoid unnecessary
recalculations.

.. code-block:: python

  from sklearn.preprocessing import Binarizer

  # Simple binarization step that we want ot cache
  binarizer = [('binarizer', Binarizer())]

  # Reskit needs to define steps in this manner
  steps = [('binarizer', binarizer),
           ('classifier', classifiers)]

  pipe = Pipeliner(steps=steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
  pipe.plan_table


.. csv-table::
  :file: pipeliner_caching0.csv


.. code-block:: python

  pipe.get_results(X, y, caching_steps=['binarizer'])


.. code-block:: bash

  Line: 1/3
  Line: 2/3
  Line: 3/3


.. csv-table::
  :file: pipeliner_caching1.csv


Last cached calculations stored in _cached_X

.. code-block:: python

  pipe._cached_X


.. code-block:: bash

  OrderedDict([('init',
                array([[-0.34004591,  0.07223225, -0.10297704, ...,  1.55809216,
                        -1.84967225,  1.20716726],
                       [-0.61534739, -0.2666859 , -1.21834152, ..., -1.31814689,
                         0.97544639, -1.21321157],
                       [ 1.08934663,  0.12345205,  0.09360395, ..., -0.50379748,
                        -0.03416718,  1.51609726],
                       ..., 
                       [-1.06428161, -0.22220536, -2.87462458, ..., -0.17236827,
                        -0.22141068,  2.76238087],
                       [ 0.40555432,  0.12063241,  1.1565546 , ...,  1.71135941,
                         0.29149897, -0.67978708],
                       [-0.47521282,  0.11614697,  0.45649735, ..., -0.15355913,
                         0.19643313,  0.67876913]])),
               ('binarizer', array([[ 0.,  1.,  0., ...,  1.,  0.,  1.],
                       [ 0.,  0.,  0., ...,  0.,  1.,  0.],
                       [ 1.,  1.,  1., ...,  0.,  0.,  1.],
                       ..., 
                       [ 0.,  0.,  0., ...,  0.,  0.,  1.],
                       [ 1.,  1.,  1., ...,  1.,  1.,  0.],
                       [ 0.,  1.,  1., ...,  0.,  1.,  1.]]))])
