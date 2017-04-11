==================
Transformers Usage
==================

This tutorial helps you to understand how you can transform your data using
DataTransformer class and how to make your own classes for data transformation.

1. Simple Example
-----------------

.. code-block:: python

  import numpy as np

  from reskit.normalizations import mean_norm
  from reskit.core import DataTransformer
  from reskit.core import walker_by_zero_dim

  matrix_0 = np.random.rand(5, 5)
  matrix_1 = np.random.rand(5, 5)
  matrix_2 = np.random.rand(5, 5)
  y = np.array([0, 0, 1])

  X = np.array([matrix_0,
                matrix_1,
                matrix_2])

  output = np.array([mean_norm(matrix_0),
                     mean_norm(matrix_1),
                     mean_norm(matrix_2)])

  result = DataTransformer(
                  global_func=walker_by_zero_dim,
                  local_func=mean_norm ).fit_transform(X)

  (output == result).all()

.. code-block:: bash

  True

DataTransformer uses two functions for transformation: global function and
local function. A global function helps to define the way a local function will
be used to transform data. A local function is a simple transformation of one
sample from the data. For example, a local function may be normalisation by
mean, as in the instance above.

We used global function walker_by_zero_dim there. As you can see from the
following source code it transforms each matrix in data according to a local
function.

.. code-block:: python

  def walker_by_zero_dim(func, X, **params):

      X = X.copy()
      new_X = []
      for i in range(len(X)):
          new_X.append(func(X[i], **params))
      return array(new_X)

If you have a data with specific data structure it is useful and convenient to
write your function for data processing.

2. Use dictionaries for data
----------------------------

In some cases, it is useful to store some additional information in X to
creation final features set X.

.. code-block:: python

  from reskit.core import walker_by_ids


  def are_dicts_equal(dict_1, dict_2):

      if dict_1.keys() != dict_2.keys():
          return False

      equal = True
      for key in dict_1:
          if (dict_1[key] != dict_2[key]).all():
              return False

      return True

  X = {'matrices': {'id1': matrix_0,
                    'id2': matrix_1,
                    'id3': matrix_2}}

  output = {'matrices': {'id1': mean_norm(matrix_0),
                         'id2': mean_norm(matrix_1),
                         'id3': mean_norm(matrix_2)}}

  result = DataTransformer(
      global_func=walker_by_ids,
      global__from_field='matrices',
      global__to_field='matrices',
      local_func=mean_norm).fit_transform(X)

  are_dicts_equal(output['matrices'], result['matrices'])

.. code-block:: bash

  True

Global and local functions can have their own parameters. To access global
function parameters you should write "global__" before a needed parameter, as
in the instance above. Other parameters you write in DataTransformer input will
be referred to local_function parameters.

3. Transform a data dictionary to an usual array
------------------------------------------------

But if we use X as dictionary we should transform it into an array before usage
in usual sklearn machine learning pipelines. Usually, we want to choose just
one field from the dictionary and use it as X array, but sometimes we want to
collect X array from different fields of the dictionary. In this case, we use
collect parameter of DataTransformer. If you put a list of fields from X
dictionary to DataTransformer, it stacks horizontally arrays from this fields
to one X array. In the following instance, we created bag_of_edges and degrees
features for our graphs and stack they for one X array.

.. code-block:: python

  from reskit.features import bag_of_edges
  from reskit.features import degrees


  degrees_features = np.array(
      [degrees(X['matrices']['id1']),
       degrees(X['matrices']['id2']),
       degrees(X['matrices']['id3'])])

  bag_of_edges_features = np.array(
      [bag_of_edges(X['matrices']['id1']),
       bag_of_edges(X['matrices']['id2']),
       bag_of_edges(X['matrices']['id3'])])

  output_X = np.hstack((degrees_features, bag_of_edges_features))

  temp_X = DataTransformer(
      global_func=walker_by_ids,
      global__from_field='matrices',
      global__to_field='degrees',
      local_func=degrees).fit_transform(X)

  result_X = DataTransformer(
      global_func=walker_by_ids,
      global__from_field='matrices',
      global__to_field='bag_of_edges',
      global__collect=['degrees', 'bag_of_edges'],
      local_func=bag_of_edges).fit_transform(temp_X)

  (result_X == output_X).all()

.. code-block:: bash

  True

4. Your own transformer
-----------------------

If you need more flexibility in transformation, you can implement your own
transformer. Simplest example:

.. code-block:: python

  from sklearn.base import TransformerMixin
  from sklearn.base import BaseEstimator

  class MyTransformer(BaseEstimator, TransformerMixin):
      
      def __init__(self):
          pass
      
      def fit(self, X, y=None, **fit_params):
          #
          # Write here the code if transformer need
          # to learn anything from data.
          #
          # Usually nothing should be here, 
          # just return self.
          #
          return self
      
      def transform(self, X):
          #
          # Write here your transformation
          #
          return X
