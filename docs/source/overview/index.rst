========
Overview
========

Reskit (researcher’s kit) is a library for creating and curating reproducible
pipelines for scientific and industrial machine learning. The natural extension
of the ``scikit-learn`` Pipelines to general classes of pipelines, Reskit
allows for the efficient and transparent optimization of each pipeline step.
Main features include data caching, compatibility with most of the scikit-learn
objects, optimization constraints (e.g. forbidden combinations), and table
generation for quality metrics. Reskit also allows for the injection of custom
metrics into the underlying scikit frameworks. Reskit is intended for use by
researchers who need pipelines amenable to versioning and reproducibility, yet
who also have a large volume of experiments to run.

.. contents::


Features
--------

  * Ability to combine pipelines with equal number of steps in list of
    experiments, running them and returning results in a convenient format for
    analysis (Pandas dataframe).

  * Preprocessing steps caching. Usual SciKit-learn pipelines cannot cache
    temporary steps. We provide an opportunity to save fixed steps, so in next
    pipeline already calculated steps won’t be recalculated.

  * Ability to set "forbidden combinations" for chosen steps of a pipeline. It
    helps to test only needed pipelines, not all possible combinations. 
  
  * Full compatibility with scikit-learn objects. It means you can use in
    Reskit any scikit-learn data transforming object or any predictive model
    implemented in scikit-learn.

  * Evaluating experiments using several performance metrics. 

  * Creation of transformers for your own tasks through DataTransformer class,
    which allows you to use your functions as data processing steps in
    pipelines. 

  * Tools for machine learning on networks, particularly, for connectomics.
    Particularly, you can normalize adjacency matrices of graphs and calculate
    state-of-the-art local metrics using DataTransformer and BCTpy (Brain
    Connectivity Toolbox python version) or use some implemented in Reskit
    metrics [3]

Getting started: A Short Introduction to Reskit
-----------------------------------------------
Let's say we want to prepare data and try some scalers and classifiers for
prediction in a classification problem. We will tune paramaters of classifiers
by grid search technique.

Data preparing:

.. code-block:: python

  from sklearn.datasets import make_classification


  X, y = make_classification()


Setting steps for our pipelines and parameters for grid search:

.. code-block:: python
  
  from reskit.core import Pipeliner

  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import MinMaxScaler

  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC


  classifiers = [('LR', LogisticRegression()),
                 ('SVC', SVC())]

  scalers = [('standard', StandardScaler()),
             ('minmax', MinMaxScaler())]

  steps = [('scaler', scalers),
           ('classifier', classifiers)]

  param_grid = {'LR': {'penalty': ['l1', 'l2']},
                'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}} 


Setting a cross-validation for grid searching of hyperparameters and for evaluation of models with obtained hyperparameters.

.. code-block:: python

  from sklearn.model_selection import StratifiedKFold


  grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


Creating a plan of our research:

.. code-block:: python

  pipeliner = Pipeliner(steps=steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
  pipeliner.plan_table 

.. csv-table::
  :file: overview_plan_table.csv


To tune parameters of models and evaluate this models, run:

.. code-block:: python

  pipeliner.get_results(X, y, scoring=['roc_auc'])


.. code-block:: bash

  Line: 1/4
  Line: 2/4
  Line: 3/4
  Line: 4/4


.. csv-table::
  :file: overview_results.csv


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

