"""
Utility functions for loading feature datasets
"""

import numpy as np

def load_csv(path):
    data = np.loadtxt(path, delimiter=",")
    X = data[:, 1:-1]
    y = data[:, -1].astype(int)
    return X, y
