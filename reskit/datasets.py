import pickle
import os

module_path = os.path.dirname(__file__)
def load_UCLA_data():
    with open(os.path.join(module_path, 'datasets', 'UCLA_data.pickle'), 'rb') as f:
        X = pickle.load(f)
    y = X.pop('y')
    return X, y
