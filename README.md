# Reskit
[![Documentation Status](https://readthedocs.org/projects/reskit/badge/?version=0.1.x)](http://reskit.readthedocs.io/en/0.1.x/?badge=0.1.x)
[![Join the chat at https://gitter.im/ResearcherKit/Lobby](https://badges.gitter.im/ResearcherKit/Lobby.svg)](https://gitter.im/ResearcherKit/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/neuro-ml/reskit.svg?branch=master)](https://travis-ci.org/neuro-ml/reskit)
[![codecov](https://codecov.io/gh/neuro-ml/reskit/branch/master/graph/badge.svg)](https://codecov.io/gh/neuro-ml/reskit)

Reskit is a Python language package for making reproducible machine learning
experiments. Mainly built on ``scikit-learn``, it helps to work with machine
learning pipelines fast and efficient.

## Goals

Reskit is intended to provide:

* simply reproducible experiments
* graph's matrix norms and metrics

## Example

Let's say we want to prepare data and try some scalers and classifiers for
prediction in a classification problem. We will tune paramaters of classifiers
by grid search technique.

Data preparing:

```python
from sklearn.datasets import make_classification


X, y = make_classification()
data = {'X': X, 'y': y}
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

Creating a plan of our research:

```python
pipeliner = Pipeliner(steps=steps, param_grid=param_grid)
pipeliner.plan_table
```

| Scaler   | Classifier |
|----------|------------|
|  minmax  |     LR     |
|  minmax  |     SVC    |
| standard |     LR     |
| standard |     SVC    |

To tune parameters of models and evaluate this models, run:

```python
pipe.get_results(data=data, scoring=['roc_auc'])
```

```bash
Line: 1/4
Line: 2/4
Line: 3/4
Line: 4/4
```
|  Scaler  | Classifier | grid_roc_auc_mean | grid_roc_auc_std | grid_roc_auc_best_params | eval_roc_auc_mean | eval_roc_auc_std |         eval_roc_auc_scores         |
|----------|------------|-------------------|------------------|--------------------------|-------------------|------------------|-------------------------------------|
|  minmax  |     LR     |      0.869559     |     0.0368628    | {'penalty': 'l1'}        |     0.869954      |     0.0368373    | [ 0.83044983 0.86029412 0.91911765] |
|  minmax  |     LR     |      0.839706     |     0.0616379    | {'kernel': 'linear'}     |     0.840254      |     0.0617057    | [ 0.78546713 0.80882353 0.92647059] |
| standard |     SVC    |      0.849007     |     0.0389125    | {'penalty: 'l1'}         |     0.849265      |     0.0390237    | [ 0.82352941 0.81985294 0.90441176] |
| standard |     SVC    |      0.839669     |     0.0565861    | {'kernel': 'sigmoid'}    |     0.840182      |     0.0566397    | [ 0.78892734 0.8125 0.91911765]     |

The documentation includes more detailed [tutorial](http://reskit.readthedocs.io/en/latest/tutorial/index.html).

## Installation

Reskit currently requires ``Python 3.4`` or later to run. Please install ``Python`` and
``pip`` via the package manager of your operating system if it is not included
already.

Reskit depends on:

* numpy
* scikit-learn
* pandas

To install dependencies run next command:

```bash
pip install -r https://raw.githubusercontent.com/neuro-ml/reskit/master/requirements.txt
```
To install the latest development version of Reskit, run the following commands:

```bash
pip install -U https://github.com/neuro-ml/reskit/archive/master.zip
```
Some reskit functions depends on:

* scipy
* python-igraph
* networkx

You may install it via:

```bash
pip install -r https://raw.githubusercontent.com/nuro-ml/reskit/master/requirements_additional.txt
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
    docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit bash
    ```

  * If you want to start Jupyter Notebook server at http://localhost:8809 in container:

    ```bash
    docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit jupyter notebook --no-browser --ip="*"
    ```

    Open http://localhost:8809 on your local machine in a web browser.
