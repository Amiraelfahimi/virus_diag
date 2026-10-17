"""Evaluator : mesure la qualité d'un modèle entraîné."""
from virus_diag.utils import metrics


class Evaluator:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def evaluate(self, X, y):
        proba = self.model.predict(X)
        y_pred = (proba >= self.threshold).astype(int)
        return {
            "accuracy": metrics.accuracy(y, y_pred),
            "precision": metrics.precision(y, y_pred),
            "recall": metrics.recall(y, y_pred),
            "f1": metrics.f1_score(y, y_pred),
            "loss": metrics.binary_cross_entropy(y, proba),
            "confusion": metrics.confusion_matrix(y, y_pred),
        }
