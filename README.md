# Reskit

[![Documentation Status](https://readthedocs.org/projects/reskit/badge/?version=0.1.0)](http://reskit.readthedocs.io/en/0.1.0/?badge=0.1.0)
[![Join the chat at https://gitter.im/ResearcherKit/Lobby](https://badges.gitter.im/ResearcherKit/Lobby.svg)](https://gitter.im/ResearcherKit/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/neuro-ml/reskit.svg?branch=master)](https://travis-ci.org/neuro-ml/reskit)
[![codecov](https://codecov.io/gh/neuro-ml/reskit/branch/master/graph/badge.svg)](https://codecov.io/gh/neuro-ml/reskit)


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

## Features

* Ability to combine pipelines with an equal number of steps in list of
  experiments, running them and returning results in a convenient format for
  human consumption (Pandas dataframe).

* Step caching. Standard SciKit-learn pipelines cannot cache temporary steps.
  Reskit includes the option  to save fixed steps, so in next pipeline
  specified steps won’t be recalculated.

* Forbidden combination constraints. Not all possible combinations of pipelines
  are viable or meaningfully different. For example, in a classification task
  comparing the performance of  logistic regression and decision trees the
  former requires feature scaling while the latter may not. In this case you
  can block the unnecessary pair. Reskit supports general tuple blocking as
  well. 

* Full compatibility with scikit-learn objects. Reskit can use any scikit-learn
  data transforming object and/or predictive model, and assumably many other
  libraries that uses the scikit template.

* Evaluation of multiple performance metrics simultaneously. Evaluation is
  simply another step in the pipeline, so we can specify a number of possible
  evaluation metrics and Reskit will expand out the computations for each
  metric for each pipeline.

* The DataTransformer class, which is Reskit’s simplfied interface for
  specifying fit/transform methods in pipeline steps. A DataTransformer
  subclass need only specify one function.

* Tools for learning on graphs. Due to our original motivations Reskit includes
  a number of operations for network data. In particular, it allows  a variety
  of normalization choices foradjacency matrices , as well as built in  local
  graph metric calculations. These were implemented using  DataTransformer and
  in some cases the BCTpy (the Brain Connectivity Toolbox python version)

## Documentation

The documentation includes detailed
[tutorial](http://reskit.readthedocs.io/en/0.1.0/tutorial/index.html), but for
a quick view, we provide for you an example.

## Example

Let's say we want to prepare data and try some scalers and classifiers for
prediction in a classification problem. We will tune paramaters of classifiers
by grid search technique.

Data preparing:

```python
from sklearn.datasets import make_classification


X, y = make_classification()
```

Setting steps for our pipelines and parameters for grid search:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

scalers = [
    ('minmax', MinMaxScaler()),
    ('standard', StandardScaler())
]

classifiers = [
    ('LR', LogisticRegression()),
    ('SVC', SVC())
]

steps = [
    ('Scaler', scalers),
    ('Classifier', classifiers)
]

param_grid = {
    'LR' : {
        'penalty' : ['l1', 'l2']},
    'SVC' : {
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}}
```

Setting a cross-validation for grid searching of hyperparameters and for
evaluation of models with obtained hyperparameters.

```python
from sklearn.model_selection import StratifiedKFold


grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
```

Creating a plan of our research:

```python
pipeliner = Pipeliner(steps=steps, grid_cv=grid_cv, eval_cv=eval_cv, param_grid=param_grid)
pipeliner.plan_table
```

  |    |  scaler    |  classifier
  |----|------------|-------------
  | 0  |  standard  |  LR
  | 1  |  standard  |  SVC
  | 2  |  minmax    |  LR
  | 3  |  minmax    |  SVC

To tune parameters of models and evaluate this models, run:

```python
pipeliner.get_results(X, y, scoring='roc_auc')
```

```bash
Line: 1/4
Line: 2/4
Line: 3/4
Line: 4/4
```

  |   |  scaler    |  classifier  |  grid_roc_auc_mean  |  grid_roc_auc_std  |  grid_roc_auc_best_params  |  eval_roc_auc_mean  |  eval_roc_auc_std  |  eval_roc_auc_scores
  |---|------------|--------------|---------------------|--------------------|----------------------------|---------------------|--------------------|---------------------------------
  | 0 |  standard  |  LR          |  0.956              |  0.0338230690506   |  {'penalty': 'l1'}         |  0.968              |  0.0324961536185   |  [ 0.92  1.    1.    0.94  0.98]
  | 1 |  standard  |  SVC         |  0.962              |  0.0278567765544   |  {'kernel': 'poly'}        |  0.976              |  0.0300665927567   |  [ 0.95  1.    1.    0.93  1.  ]
  | 2 |  minmax    |  LR          |  0.964              |  0.0412795348811   |  {'penalty': 'l1'}         |  0.966              |  0.0377359245282   |  [ 0.92  1.    1.    0.92  0.99]
  | 3 |  minmax    |  SVC         |  0.958              |  0.0411825205639   |  {'kernel': 'rbf'}         |  0.962              |  0.0401995024845   |  [ 0.93  1.    1.    0.9   0.98]


## Installation

Reskit currently requires ``Python 3.4`` or later to run. Please install
``Python`` and ``pip`` via the package manager of your operating system if it
is not included already.

Reskit depends on:

* `numpy`
* `scikit-learn`
* `pandas`
* `scipy`
* `python-igraph`
* `networkx`

If you don't want to use graph metrics, you should to comment `scipy`,
`python-igraph` and `networkx` in `requirements.txt`.

To install dependencies run next command:

```bash
pip install -r https://raw.githubusercontent.com/neuro-ml/reskit/master/requirements.txt
```

Last stable version is 0.1.0. You can install it via:

```bash
pip3 install -U https://github.com/neuro-ml/reskit/archive/0.1.0.zip
```

To install the latest development version of Reskit, run the following commands:

```bash
pip install -U https://github.com/neuro-ml/reskit/archive/master.zip
```

## Docker

If you just want to try Reskit or don’t want to install Python, you can build
docker image and make all reskit’s stuff there. Also, in this case, you can
provide the simple way to reproduce your experiment. To run Reskit in docker
you can use next commands.

1. Clone repository:

    ```bash
    git clone https://github.com/neuro-ml/reskit.git
    cd reskit
    ```

2. Build docker image:

    ```bash
    docker build -t docker-reskit -f Dockerfile .
    ```

3. Run docker image.
  * If you want to run bash in container:

    ```bash
    docker run -it docker-reskit bash
    ```

  * If you want to run bash in container with shared directory:

    ```bash
    docker run -v $PWD/scripts:/reskit/scripts -it docker-reskit bash
    ```

  * If you want to start Jupyter Notebook server at http://localhost:8809 in
    container:

    ```bash
    docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8809 docker-reskit jupyter notebook --no-browser --ip="*" --allow-root --port 8809
    ```
    
    You will see message:

    ```bash
      Copy/paste this URL into your browser when you connect for the first time,
      to login with a token:
        http://localhost:8809/?token=some_token
    ```
