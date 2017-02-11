========
Overview
========
Reskit is a Python language package for making reproducible machine learning experiments. 
Mainly built on ``scikit-learn`` it helps to work with machine learning pipelines fast and efficient. 

Goals
-----

Reskit is intended to provide
  * makes possible simply reproducible experiments
  * graph's matrix norms and metrics

Getting started: 30 seconds to Reskit
-------------------------------------

Let's say we have two datasets, and we want to compare some pipelines on
this datasets. Machine learning pipelines represent a list of transforms
and a final estimator. For instance, we want to try two different
classifiers and two different scale methods.

We define functions of loading datasets. 
This functions should return a dictionary with fields `X` and `y`. 
For this example we use scikit-learn function for making classification problem.

.. code-block:: python

	from sklearn.datasets import make_classification

	def get_dataset1(path):
		X, y = make_classification(n_samples=200, n_features=30, n_informative=10)
		return { 'X' : X, 'y' : y }

	def get_dataset2(path):
		X, y = make_classification(n_samples=300, n_features=40, n_informative=15)
		return { 'X' : X, 'y' : y }

Because of transforms of the pipeline should have methods ``fit`` and ``transform``, 
we provide class Transformer, that allows using functions in a pipeline. 
``collect`` needs for taking ``X`` and ``y`` from the dictionary for putting (X, y) tuple to next transform.

.. code-block:: python

	from reskit.core import Transformer


	datasets = [
		('dataset1', Transformer(get_dataset1, collect=['X'])),
		('dataset2', Transformer(get_dataset2, collect=['X']))
	]

Firstly, we use feature selection method ``VarianceThreshold``.

.. code-block:: python

	from sklearn.feature_selection import VarianceThreshold


	selectors = [
		('var_threshold', VarianceThreshold())
	]

Next, we use one of two scaling transformations.

.. code-block:: python

	from sklearn.preprocessing import StandardScaler
	from sklearn.preprocessing import MinMaxScaler


	scalers = [
		('minmax', MinMaxScaler()),
		('standard', StandardScaler())
	]

In the end, we use one of two classifiers.

.. code-block:: python

	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import SVC


	classifiers = [
		('LR', LogisticRegression()),
		('SVC', SVC())
	]

For grid search of hyperparameters and for evaluation of results we use stratified cross validation.

.. code-block:: python

	from sklearn.model_selection import StratifiedKFold


	grid_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
	eval_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

Let's set `Pipeliner` class and look at the plan table of our experiment.

.. code-block:: python

	from reskit.core import Pipeliner

	steps = [
		('Data', datasets),
		('Selector', selectors),
		('Scaler', scalers),
		('Classifier', classifiers)
	]

	param_grid = {
		'LR' : {
			'penalty' : ['l1', 'l2']
		},
		'SVC' : {
			'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
		}
	}

	pipe = Pipeliner(steps, eval_cv=eval_cv, grid_cv=grid_cv, param_grid=param_grid)
	pipe.plan_table

+---+----------+---------------+------------+----------------+
|   | **Data** | **Selector**  | **Scaler** | **Classifier** |
+---+----------+---------------+------------+----------------+
| 0 | dataset1 | var_threshold |   minmax   |       LR       | 
+---+----------+---------------+------------+----------------+
| 1 | dataset1 | var_threshold |   minmax   |       SVC      |
+---+----------+---------------+------------+----------------+
| 2 | dataset1 | var_threshold |  standard  |       LR       |
+---+----------+---------------+------------+----------------+
| 3 | dataset1 | var_threshold |  standard  |       SVC      | 
+---+----------+---------------+------------+----------------+
| 4 | dataset2 | var_threshold |   minmax   |       LR       |
+---+----------+---------------+------------+----------------+
| 5 | dataset2 | var_threshold |   minmax   |       SVC      |
+---+----------+---------------+------------+----------------+
| 6 | dataset2 | var_threshold |  standard  |       LR       |
+---+----------+---------------+------------+----------------+
| 7 | dataset2 | var_threshold |  standard  |       SVC      |
+---+----------+---------------+------------+----------------+

The plan is OK. To get results we run:

.. code-block:: python

	pipe.get_results(data='path/to/directory', caching_steps=['Data'], scoring=['roc_auc'])

.. code-block:: bash

  Line: 1/8
  Line: 2/8
  Line: 3/8
  Line: 4/8
  Line: 5/8
  Line: 6/8
  Line: 7/8
  Line: 8/8

+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
|   | **Data** | **Selector**  | **Scaler** | **Classifier** | **grid_roc_auc_mean** | **grid_roc_auc_std** | **grid_roc_auc_best_params** | **eval_roc_auc_mean** | **eval_roc_auc_std** |       **eval_roc_auc_scores**       |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 0 | dataset1 | var_threshold |   minmax   |       LR       |       0.958583        |      0.00380304      |       {'penalty': 'l1'}      |        0.942676       |       0.016551       | [ 0.95934256 0.94857668 0.92011019] |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 1 | dataset1 | var_threshold |   minmax   |       SVC      |       0.938877        |      0.0221093       |      {'kernel': 'linear'}    |        0.91035        |       0.0306385      | [ 0.93858131 0.92470156 0.8677686 ] |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 2 | dataset1 | var_threshold |  standard  |       LR       |       0.955178        |      0.0133661       |       {'penalty': 'l1'}      |        0.920474       |       0.0271148      | [ 0.95242215 0.92286501 0.88613407] |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 3 | dataset1 | var_threshold |  standard  |       SVC      |       0.959082        |      0.0123213       |       {'kernel': 'rbf'}      |        0.925589       |       0.0226491      | [ 0.9567474 0.9164371 0.90358127]   |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 4 | dataset2 | var_threshold |   minmax   |       LR       |       0.808662        |      0.0159014       |       {'penalty': 'l1'}      |        0.803197       |       0.024606       | [ 0.82078431 0.7684 0.82040816]     |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 5 | dataset2 | var_threshold |   minmax   |       SVC      |       0.779896        |      0.0127644       |      {'kernel': 'linear'}    |        0.762861       |       0.0230988      | [ 0.78431373 0.7308 0.77346939]     |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 6 | dataset2 | var_threshold |  standard  |       LR       |       0.797357        |      0.0094159       |       {'penalty': 'l1'}      |        0.769821       |       0.0236991      | [ 0.80313725 0.75 0.75632653]       |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+
| 7 | dataset2 | var_threshold |  standard  |       SVC      |       0.873096        |      0.0213294       |       {'kernel': 'rbf'}      |        0.847338       |       0.0261201      | [ 0.8654902 0.8104 0.86612245]      |
+---+----------+---------------+------------+----------------+-----------------------+----------------------+------------------------------+-----------------------+----------------------+-------------------------------------+

Installation
------------

Reskit currently requires ``Python 3.4`` or later to run.
Please install ``Python`` and ``pip`` via the package manager of your operating system if it is not included already.

Reskit depends on:
  - `numpy <http://www.numpy.org/>`_
  - `scikit-learn <http://scikit-learn.org/stable/>`_
  - `pandas <http://pandas.pydata.org/>`_


To install dependencies run next command:

.. code-block:: bash

	pip install -r https://raw.githubusercontent.com/neuro-ml/reskit/master/requirements.txt

To install stable version, run the following command:

.. code-block:: bash

	pip install -U https://github.com/neuro-ml/reskit/archive/master.zip

To install latest development version of Reskit, run the following commands:

.. code-block:: bash

  pip install https://github.com/neuro-ml/reskit/archive/master.zip

Some reskit functions depends on:

  - `scipy <https://www.scipy.org/>`_
  - `python-igraph <http://igraph.org/python/>`_
  - `networkx <https://networkx.github.io/>`_

You may install it via:

.. code-block:: bash

  pip install -r https://raw.githubusercontent.com/nuro-ml/reskit/master/requirements_additional.txt

Docker
------

If you just want to try Reskit or don't want to install Python, 
you can build docker image and make all reskit's stuff there. 
Also, in this case, you can provide the simple way to reproduce your experiment.
To run Reskit in docker you can use next commands.

1. Clone:

.. code-block:: bash

  git clone https://github.com/neuro-ml/reskit.git
  cd reskit

2. Build:

.. code-block:: bash

  docker build -t docker-reskit -f Dockerfile .

3. Run container.

  a) If you want to run bash in container:

  .. code-block:: bash

    docker run -it docker-reskit bash

  b) If you want to run bash in container with shared directory:

    .. code-block:: bash

      docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit bash

    .. note:: 
      
      Files won't be deleted after stopping container if you save this
      files in shared directory.

  c) If you want to start Jupyter Notebook server at ``http://localhost:8809`` in container:

    .. code-block:: bash

      docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit jupyter notebook --no-browser --ip="*"

    Open http://localhost:8809 on your local machine in a web browser.

