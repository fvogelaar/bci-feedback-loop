"""
Offline training script
"""

from src.dataset import load_csv
from src.classifier import OnlineLDA

X, y = load_csv("features_labeled.csv")

clf = OnlineLDA()
clf.train(X, y)

print("Model trained.")
