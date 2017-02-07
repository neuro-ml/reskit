============
Installation
============

Prerequisites
-------------

Reskit currently requires ``Python 3.4`` or later to run.
Please install ``Python`` and ``pip`` via the package manager of your operating system if it is not included already.

Make sure you installed prerequisites for packages:

  - `numpy <http://www.numpy.org/>`_
  - `scipy <https://www.scipy.org/>`_
  - `scikit-learn <http://scikit-learn.org/stable/>`_
  - `python-igraph <http://igraph.org/python/>`_
  - `networkx <https://networkx.github.io/>`_
  - `pandas <http://pandas.pydata.org/>`_

.. note:: If you use ``Ubuntu 16.04``, you should install packages ``libxslt1-dev`` and ``zlib1g-dev`` for ``python-igraph``.

.. Stable Reskit release
.. ---------------------

.. Reskit setup all requirements automatically. 
.. To install stable version, run the following command:

.. .. code-block:: bash

..     pip install https://github.com/neuro-ml/reskit/archive/master.zip

Bleeding-edge version
---------------------

To install lastest development version of Reskit, run the following command:

.. code-block:: bash

  pip install https://github.com/neuro-ml/reskit/archive/master.zip

Docker
------

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

    .. note:: You can save files in shared directory and files won't be deleted after stopping container.

    .. code-block:: bash

      docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit bash

  c) If you want to start Jupyter Notebook server at ``http://localhost:8809`` in container:

    .. code-block:: bash

      docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit jupyter notebook --no-browser --ip="*"

    Open http://localhost:8809 on your local machine in a web browser.
