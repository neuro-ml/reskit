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
  from reskit.core import MatrixTransformer


  normalized_X = MatrixTransformer(
      func=mean_norm).fit_transform(X)

  featured_X = MatrixTransformer(
      func=bag_of_edges).fit_transform(normalized_X)


3. Brain Connectivity Toolbox
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

  featured_X = MatrixTransformer(
      d=0.85,
      func=pagerank_centrality).fit_transform(X)


If we want to try pagerank_centrality and degrees for SVM and
LogisticRegression classfiers.

.. code-block:: python

  from bct.algorithms.degree import degrees_und

  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  from sklearn.model_selection import StratifiedKFold

  from reskit.core import Pipeliner

  # Feature extraction step variants (1st step)
  featurizers = [('pagerank', MatrixTransformer(    
                                  d=0.85,
                                  func=pagerank_centrality)),
                 ('degrees', MatrixTransformer(
                                  func=degrees_und))]

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

  # Setting cross-validations
  grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

  pipe = Pipeliner(steps=steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
  pipe.plan_table


.. csv-table::
  :file: ml_graphs_results.csv


.. code-block:: python

  pipe.get_results(X, y, scoring=scoring, caching_steps=['featurizer'])


.. code-block:: bash

  Line: 1/4
  Line: 2/4
  Line: 3/4
  Line: 4/4

  
.. csv-table::
  :file: ml_graphs_results.csv


This is the main things about maching learning on graphs. Now you can try big
amount of normalizations features and classifiers for graphs classifcation. In
case you need something specific you can implement temporary pipeline step to
fiegure out the influence of this step on the result.
