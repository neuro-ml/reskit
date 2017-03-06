=======================
Transformer Class Usage
=======================

1. Transformer Data Format
--------------------------

To use your data in Reskit, you should transform it to a dictionary with
fields ``X`` and ``y``. Here, we use a scikit-learn function for making
classification problem.

.. code-block:: python

  from sklearn.datasets import make_classification

  X, y = make_classification()
  data = {'X': X, 'y': y}

It's agreement of Reskit and you can see advantages of this approach in
Use Cases section.

2. Functions Usage in Through Transformer
-----------------------------------------

To create transformation of the max norm to first feature column, run: 

.. code-block:: python

	from reskit.core import Transformer
	import numpy as np

	def first_feature_max_norm(data):
			first_feature_max = np.max(data['X'][:,0])
			data['X'][:,0] = data['X'][:,0] / first_feature_max
			return data

	transformer = Transformer(func=first_feature_max_norm)

To make the transformation, run:

.. code-block:: python

	temp_data = transformer.fit_transform(data)

3. Transformation Data to (X, y) Form
-------------------------------------

To use transformed data in ``scikit-learn`` pipelines use ``collect`` parameter of Transformer 
In case you define ``collect``, Transformer returns dictionary field that defined in ``collect`` parameter and ``y`` field.

.. code-block:: python

	transformer = Transformer(func=mean_norm, collect='X')
	X, y = transformer.fit_transform(data)
