============
Installation
============

Prerequisites
-------------

Reskit currently requires ``Python 3.4`` to run.
Please install ``Python`` and ``pip`` via the package manager of your operating system if it is not included already.

Make sure you installed prerequisites for packages:

  - `numpy <http://www.numpy.org/>`_
  - `scipy <https://www.scipy.org/>`_
  - `scikit-learn <http://scikit-learn.org/stable/>`_
  - `python-igraph <http://igraph.org/python/>`_
  - `networkx <https://networkx.github.io/>`_
  - `pandas <http://pandas.pydata.org/>`_

Stable Lasagne release
----------------------

Reskit setup all requirements automatically. 
To install stable version, run the following command:

.. code-block:: bash

  will be soon

Bleeding-edge version
---------------------

To install lastest development version of Reskit, run the following command:

.. code-block:: bash

  pip install https://github.com/neuro-ml/reskit/archive/master.zip

Docker
------

To run Reskit into docker you can use next commands.

1. Clone:

.. code-block:: bash

  git clone https://github.com/neuro-ml/reskit.git
  cd reskit

2. Build:

.. code-block:: bash

  docker build -t docker-reskit -f Dockerfile .

3.a Run container in bash:

.. code-block:: bash

  docker run -it docker-reskit bash

3.b Run container in bash with shared directory (you can save files in this directory and it won't be deleted with stopping container).

.. code-block:: bash

  docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit bash

3.c Run container and Jupyter Notebook server at ``http://localhost:8809``. 
  Open http://localhost:8809 on your local machine in a web browser.

.. code-block:: bash

  docker run -v $PWD/scripts:/reskit/scripts -it -p 8809:8888 docker-reskit bash -с 'jupyter notebook --no-browser --ip="*"'
