========
Overview
========
Reskit is a Python language package for making reproducible machine learning experiments. 
Mainly built on ``scikit-learn`` it helps to work with machine learning pipelines fast and efficient. 

Goals
-----

Reskit is intended to provide
  * fast and efficient work with lots of machine learning pipelines.
  * graph's matrix norms and metrics
  * makes possible simply reproducible experiments
  * simple support of self-implemented transformers in pipelines

Installation
------------

Reskit currently requires ``Python 3.4`` or later to run.
Please install ``Python`` and ``pip`` via the package manager of your operating system if it is not included already.

Reskit depends on:
  - `numpy <http://www.numpy.org/>`_
  - `scipy <https://www.scipy.org/>`_
  - `scikit-learn <http://scikit-learn.org/stable/>`_
  - `python-igraph <http://igraph.org/python/>`_
  - `networkx <https://networkx.github.io/>`_
  - `pandas <http://pandas.pydata.org/>`_

To install dependencies run next command:

.. code-block:: bash

	pip install -r https://raw.githubusercontent.com/neuro-ml/reskit/master/requirements.txt

.. note:: 

  If you use ``Ubuntu 16.04``, you should install packages ``libxslt1-dev`` and ``zlib1g-dev`` for ``python-igraph``.

To install stable version, run the following command:

.. code-block:: bash

	pip install -U https://github.com/neuro-ml/reskit/archive/master.zip

To install latest development version of Reskit, run the following commands:

.. code-block:: bash

  pip install https://github.com/neuro-ml/reskit/archive/master.zip

Getting started: 30 seconds to Reskit
-------------------------------------

Wow gettign started examples.

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

