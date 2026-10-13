"""
Online-safe classifier wrapper
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

class OnlineLDA:
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, x):
        if not self.trained:
            return None
        return int(self.model.predict([x])[0])

    # Addition: crossvalidation
    def evaluate_lda_cv(X, y, n_splits=5, random_state=42):
        model = self.model
        # cross validation setup for class balance
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
            )
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        print(f"Cross-Validation Scores ({n_splits}-Fold): {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f} (± {scores.std() * 2:.4f})")

        return scores

