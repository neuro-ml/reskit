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

  pipe = Pipeliner(steps, param_grid=param_grid)
  pipe.plan_table

+----+-------------------------+------------+----------------+
|    | **feature_engineering** | **scaler** | **classifier** |
+----+-------------------------+------------+----------------+
| 0  | VT                      | standard   | LR             |
+----+-------------------------+------------+----------------+
| 1  | VT                      | standard   | SVC            |
+----+-------------------------+------------+----------------+
| 2  | VT                      | standard   | SGD            |
+----+-------------------------+------------+----------------+
| 3  | VT                      | minmax     | LR             |
+----+-------------------------+------------+----------------+
| 4  | VT                      | minmax     | SVC            |
+----+-------------------------+------------+----------------+
| 5  | VT                      | minmax     | SGD            |
+----+-------------------------+------------+----------------+
| 6  | PCA                     | standard   | LR             |
+----+-------------------------+------------+----------------+
| 7  | PCA                     | standard   | SVC            |
+----+-------------------------+------------+----------------+
| 8  | PCA                     | standard   | SGD            |
+----+-------------------------+------------+----------------+
| 9  | PCA                     | minmax     | LR             |
+----+-------------------------+------------+----------------+
| 10 | PCA                     | minmax     | SVC            |
+----+-------------------------+------------+----------------+
| 11 | PCA                     | minmax     | SGD            |
+----+-------------------------+------------+----------------+

2. Forbidden combinations
-------------------------

In case you don't want to use minmax scaler with SVC, you can define banned
combo:

.. code-block:: python
  banned_combos = [('minmax', 'SVC')]
  pipe = Pipeliner(steps, param_grid=param_grid, banned_combos=banned_combos)
  pipe.plan_table

+----+-------------------------+------------+----------------+
|    | **feature_engineering** | **scaler** | **classifier** |
+----+-------------------------+------------+----------------+
| 0  | VT                      | standard   | LR             |
+----+-------------------------+------------+----------------+
| 1  | VT                      | standard   | SVC            |
+----+-------------------------+------------+----------------+
| 2  | VT                      | standard   | SGD            |
+----+-------------------------+------------+----------------+
| 3  | VT                      | minmax     | LR             |
+----+-------------------------+------------+----------------+
| 4  | VT                      | minmax     | SGD            |
+----+-------------------------+------------+----------------+
| 5  | PCA                     | standard   | LR             |
+----+-------------------------+------------+----------------+
| 6  | PCA                     | standard   | SVC            |
+----+-------------------------+------------+----------------+
| 7  | PCA                     | standard   | SGD            |
+----+-------------------------+------------+----------------+
| 8  | PCA                     | minmax     | LR             |
+----+-------------------------+------------+----------------+
| 9  | PCA                     | minmax     | SGD            |
+----+-------------------------+------------+----------------+

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

+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
|    | **feature_engineering** | **scaler** | **classifier** | **grid_roc_auc_mean** | **grid_roc_auc_std** | **grid_roc_auc_best_params**               | **eval_roc_auc_mean** | **eval_roc_auc_std** | **eval_roc_auc_scores**             |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 0  | VT                      | standard   | LR             | 0.919338              | 0.0261305            | {'penalty': 'l1'}                          | 0.919118              | 0.0261693            | [ 0.94117647 0.93382353 0.88235294] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 1  | VT                      | standard   | SVC            | 0.913824              | 0.0230978            | {'kernel': 'rbf'}                          | 0.914072              | 0.0230813            | [ 0.88927336 0.94485294 0.90808824] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 2  | VT                      | standard   | SGD            | 0.919154              | 0.0567916            | {'l1_ratio': 0.2, 'penalty': 'elasticnet'} | 0.90953               | 0.0517625            | [ 0.93079585 0.95955882 0.83823529] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 3  | VT                      | minmax     | LR             | 0.948419              | 0.0108005            | {'penalty': 'l1'}                          | 0.948457              | 0.0108482            | [ 0.94463668 0.96323529 0.9375 ]    |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 4  | VT                      | minmax     | SGD            | 0.918382              | 0.0328705            | {'l1_ratio': 0.3, 'penalty': 'elasticnet'} | 0.924524              | 0.0260708            | [ 0.91695502 0.95955882 0.89705882] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 5  | PCA                     | standard   | LR             | 0.905074              | 0.0189125            | {'penalty': 'l1'}                          | 0.904988              | 0.0189888            | [ 0.91349481 0.92279412 0.87867647] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 6  | PCA                     | standard   | SVC            | 0.873309              | 0.0484839            | {'kernel': 'sigmoid'}                      | 0.872693              | 0.0483377            | [ 0.93425606 0.86764706 0.81617647] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 7  | PCA                     | standard   | SGD            | 0.869044              | 0.0453559            | {'l1_ratio': 0.3, 'penalty': 'elasticnet'} | 0.822376              | 0.0499817            | [ 0.87889273 0.83088235 0.75735294] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 8  | PCA                     | minmax     | LR             | 0.905515              | 0.0383624            | {'penalty': 'l1'}                          | 0.905854              | 0.0384066            | [ 0.87197232 0.95955882 0.88602941] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+
| 9  | PCA                     | minmax     | SGD            | 0.90511               | 0.0138974            | {'l1_ratio': 0.2, 'penalty': 'elasticnet'} | 0.909674              | 0.0168318            | [ 0.92387543 0.91911765 0.88602941] |
+----+-------------------------+------------+----------------+-----------------------+----------------------+--------------------------------------------+-----------------------+----------------------+-------------------------------------+

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

  pipe = Pipeliner(steps, param_grid=param_grid)
  pipe.plan_table

+---+---------------+----------------+
|   | **binarizer** | **classifier** |
+---+---------------+----------------+
| 0 | binarizer     | LR             |
+---+---------------+----------------+
| 1 | binarizer     | SVC            |
+---+---------------+----------------+
| 3 | binarizer     | SGD            |
+---+---------------+----------------+

.. code-block:: python

  pipe.get_results(X, y, caching_steps=['binarizer'])

.. code-block:: bash

  Line: 1/3
  Line: 2/3
  Line: 3/3

+---+---------------+----------------+------------------------+-----------------------+--------------------------------------------+------------------------+-----------------------+-------------------------------------+
|   | **binarizer** | **classifier** | **grid_accuracy_mean** | **grid_accuracy_std** | **grid_accuracy_best_params**              | **eval_accuracy_mean** | **eval_accuracy_std** | **eval_accuracy_scores**            |
+---+---------------+----------------+------------------------+-----------------------+--------------------------------------------+------------------------+-----------------------+-------------------------------------+
| 0 | binarizer     | LR             | 0.89                   | 0.0134774             | {'penalty': 'l1'}                          | 0.890077               | 0.0135232             | [ 0.88235294 0.87878788 0.90909091] |
+---+---------------+----------------+------------------------+-----------------------+--------------------------------------------+------------------------+-----------------------+-------------------------------------+
| 1 | binarizer     | SVC            | 0.88                   | 0.0246762             | {'kernel': 'linear'}                       | 0.879976               | 0.0247993             | [ 0.88235294 0.84848485 0.90909091] |
+---+---------------+----------------+------------------------+-----------------------+--------------------------------------------+------------------------+-----------------------+-------------------------------------+
| 3 | binarizer     | SGD            | 0.84                   | 0.0273095             | {'l1_ratio': 0.1, 'penalty': 'elasticnet'} | 0.829768               | 0.076014              | [ 0.85294118 0.72727273 0.90909091] |
+---+---------------+----------------+------------------------+-----------------------+--------------------------------------------+------------------------+-----------------------+-------------------------------------+

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
