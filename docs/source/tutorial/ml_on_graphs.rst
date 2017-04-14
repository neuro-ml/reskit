==========================
Machine Learning on Graphs
==========================

We already used some graph metrics in the previous tutorial. There we will
cover graphs metrics and features in details. Also, we will cover Brain
Connectivity Toolbox usage.

1. Realworld dataset
--------------------

Here we use UCLA autism dataset publicly available at the UCLA Multimodal
Connectivity Database. Data includes DTI-based connectivity matrices of 51
high-functioning ASD subjects (6 females) and 43 TD subjects (7 females).

.. code-block:: python

  from reskit.datasets import load_UCLA_data


  X, y = load_UCLA_data()
  X = X['matrices']

2. Normalizations and Graph Metrics
-----------------------------------

We can normalize and build some metrics.

.. code-block:: python

  from reskit.normalizations import mean_norm
  from reskit.features import bag_of_edges
  from reskit.core import DataTransformer
  from reskit.core import walker_by_zero_dim


  normalized_X = DataTransformer(
      global_func=walker_by_zero_dim,
      local_func=mean_norm).fit_transform(X)

  featured_X = DataTransformer(
      global_func=walker_by_zero_dim,
      local_func=mean_norm).fit_transform(normalized_X)

4. Brain Connectivity Toolbox
-----------------------------

We provide some basic graph metrics in Reskit. To access most state of the art
graph metrics you can use Brain Connectivity Toolbox. You should install it via
pip:

.. code-block:: bash

  sudo pip install bctpy

Let's calculate pagerank centrality of a random graph using BCT python library.

.. code-block:: python

  from bct.algorithms.centrality import pagerank_centrality
  import numpy as np


  pagerank_centrality(np.random.rand(3,3), d=0.85)

.. code-block:: bash

  array([ 0.46722034,  0.33387522,  0.19890444])

Now we calculates this metric for UCLA dataset. d is the pagerank_centrality
parameter, called damping factor (see bctpy documentation for more info). 

.. code-block:: python 

  featured_X = DataTransformer(
      global_func=walker_by_zero_dim,
      d=0.85,
      local_func=pagerank_centrality).fit_transform(X)

If we want to try pagerank_centrality and degrees for SVM and
LogisticRegression classfiers.

.. code-block:: python

  from bct.algorithms.degree import degrees_und

  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC

  from reskit.core import Pipeliner

  # Feature extraction step variants (1st step)
  featurizers = [('pagerank', DataTransformer(    
                                  global_func=walker_by_zero_dim,
                                  d=0.85,
                                  local_func=pagerank_centrality)),
                 ('degrees', DataTransformer(
                                  global_func=walker_by_zero_dim,
                                  local_func=degrees_und))]

  # Models (3rd step)
  classifiers = [('LR', LogisticRegression()),
                 ('SVC', SVC())]

  # Reskit needs to define steps in this manner
  steps = [('featurizer', featurizers),
           ('classifier', classifiers)]

  # Grid search parameters for our models
  param_grid = {'LR': {'penalty': ['l1', 'l2']},
                'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}}

  # Quality metric that we want to optimize
  scoring='roc_auc'

  pipe = Pipeliner(steps, param_grid=param_grid)
  pipe.plan_table

+---+----------------+----------------+
|   | **featurizer** | **classifier** |
+---+----------------+----------------+
| 0 | pagerank       | LR             |
+---+----------------+----------------+
| 1 | pagerank       | SVC            |
+---+----------------+----------------+
| 2 | degrees        | LR             |
+---+----------------+----------------+
| 3 | degrees        | SVC            |
+---+----------------+----------------+

.. code-block:: python

  pipe.get_results(X, y, scoring=scoring, caching_steps=['featurizer'])

.. code-block:: bash

  Line: 1/4
  Line: 2/4
  Line: 3/4
  Line: 4/4

+---+----------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+------------------------------------+
|   | **featurizer** | **classifier** | **grid_roc_auc_mean** | **grid_roc_auc_std** | **grid_roc_auc_best_params** | **eval_roc_auc_mean** | **eval_roc_auc_std** | **eval_roc_auc_scores**            |
+---+----------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+------------------------------------+
| 0 | pagerank       | LR             | 0.5                   | 0                    | {'penalty': 'l1'}            | 0.5                   | 0                    | [ 0.5 0.5 0.5]                     |
+---+----------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+------------------------------------+
| 1 | pagerank       | SVC            | 0.523565              | 0.049125             | {'kernel': 'rbf'}            | 0.523249              | 0.0492934            | [ 0.55294118 0.56302521 0.45378151]|
+---+----------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+------------------------------------+
| 2 | degrees        | LR             | 0.5346                | 0.0167932            | {'penalty': 'l2'}            | 0.53436               | 0.016723             | [ 0.55686275 0.51680672 0.52941176]|
+---+----------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+------------------------------------+
| 3 | degrees        | SVC            | 0.552512              | 0.00940143           | {'kernel': 'poly'}           | 0.552381              | 0.00936597           | [ 0.56470588 0.55042017 0.54201681]|
+---+----------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+------------------------------------+

This is the main things about maching learning on graphs. Now you can try big
amount of normalizations features and classifiers for graphs classifcation. In
case you need something specific you can implement temporary pipeline step to
fiegure out the influence of this step on the result.
