========
Tutorial
========

Motivation
----------

A central task in machine learning and data science is the comparison and
selection of models. The evaluation of a single model is very simple, and can
be carried out in a reproducible fashion using the standard scikit pipeline.
Organizing the evaluation of a large number of models is tricky; while there
are no real theory problems present, the logistics and coordination can be
tedious. Evaluating a continuously growing zoo of models is thus an even more
painful task. Unfortunately, this last case is also quite common.

``Reskit`` is a Python library that helps researchers manage this problem.
Specifically, it automates the process of choosing the best pipeline, i.e.
choosing the best set of data transformations and classifiers/regressors.
The core of reskit is two classes: ``Pipeliner`` and ``Transformer``.

First and second sections describe work of this classes.
The third section explains how to use this classes for machine learning on graphs.

You can view all tutorials in format of jupyter notebooks `here <https://github.com/neuro-ml/reskit/tree/development/tutorials>`_.

.. toctree::

  pipeliner
  transformers
