"""Régression logistique écrite avec numpy (descente de gradient)."""
import numpy as np

from virus_diag.core.model import BaseModel
from virus_diag.core.optimizer import GradientDescent
from virus_diag.utils.preprocessing import Standardizer, to_array


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionModel(BaseModel):
    def __init__(self, optimizer=None, epochs=500):
        self.optimizer = optimizer or GradientDescent(lr=0.1)
        self.epochs = epochs
        self.scaler = Standardizer()
        self.params = None

    def fit(self, X, y):
        X = self.scaler.fit_transform(to_array(X))
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self.params = {"w": np.zeros(d), "b": np.zeros(())}
        for _ in range(self.epochs):
            error = sigmoid(X @ self.params["w"] + self.params["b"]) - y
            grads = {"w": X.T @ error / n, "b": np.asarray(error.mean())}
            self.optimizer.update(self.params, grads)
        return self

    def predict(self, X):
        if self.params is None:
            raise RuntimeError("Le modèle n'est pas entraîné : appelez fit() d'abord.")
        X = self.scaler.transform(to_array(X))
        return sigmoid(X @ self.params["w"] + self.params["b"])
