[![Documentation Status](https://readthedocs.org/projects/reskit/badge/?version=latest)](http://reskit.readthedocs.io/en/latest/?badge=latest)
[![Join the chat at https://gitter.im/ResearcherKit/Lobby](https://badges.gitter.im/ResearcherKit/Lobby.svg)](https://gitter.im/ResearcherKit/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/neuro-ml/reskit.svg?branch=master)](https://travis-ci.org/neuro-ml/reskit)
[![codecov](https://codecov.io/gh/neuro-ml/reskit/branch/master/graph/badge.svg)](https://codecov.io/gh/neuro-ml/reskit)

# Reskit


Reskit is researcher kit for reproducible machine learning experiments.

# Installation

## Prerequisites

Reskit currently requires ``Python 3.4`` or later to run. 
Please install ``Python`` and ``pip`` via the package manager
or your operating system if it is not included already.

## Dependencies

Reskit automaticall install all dependencies. 
So make sure you installed prerequisites for this packages:

* numpy
* scipy
* scikit-learn
* python-igraph
* networkx
* pandas


## Bleeding-edge version

To install latest development version of Reskit, run the following command:

```bash
pip install https://github.com/neuro-ml/reskit/archive/master.zip
```

## Docker

To run Reskit in docker you can use next commands.

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
