==================
Transformers Usage
==================

This tutorial helps you to understand how you can transform your data using
DataTransformer and MatrixTransformer classes and how to make your own classes
for data transformation.

1. MatrixTransformer
--------------------

.. code-block:: python

  import numpy as np

  from reskit.normalizations import mean_norm
  from reskit.core import MatrixTransformer

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

  result = MatrixTransformer(
              func=mean_norm).fit_transform(X)

  (output == result).all()

.. code-block:: bash

  True

This is a simple example of MatrixTransformer usage. Input X for transformation
with MatrixTransformer should be a 3 dimensional array (array of matrices). So,
MatrixTransformer just transforms each matrix in X.

If you have a data with specific data structure it is useful and convenient to
write your function for data processing.

2. DataTransformer
------------------

To simply write new transformers we provide DataTransformer. The main idea is
to write functions which takes some X and output transformed X. Thus, you
shouldn't write a transformation class for compatibility with sklearn
pipelines. So, here is example of DataTransformer usage:

.. code-block:: python

  from reskit.core import DataTransformer


  def mean_norm_trans(X):
      X = X.copy()
      N = len(X)
      for i in range(N):
          X[i] = mean_norm(X[i])
      return X

  result = DataTransformer(
              func=mean_norm_trans).fit_transform(X)

  (output == result).all()

.. code-block:: bash

  True

As you can see, we writed the same transformation, but with DataTransformer
instead of MatrixTransformer.

3. Your own transformer
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
