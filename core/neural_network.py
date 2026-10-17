"""Petit réseau de neurones (1 couche cachée) écrit avec numpy."""
import numpy as np

from virus_diag.core.model import BaseModel
from virus_diag.core.optimizer import Adam
from virus_diag.core.logistic_regression import sigmoid
from virus_diag.utils.preprocessing import Standardizer, to_array


class NeuralNetworkModel(BaseModel):
    def __init__(self, hidden=8, optimizer=None, epochs=300, seed=42):
        self.hidden = hidden
        self.optimizer = optimizer or Adam(lr=0.02)
        self.epochs = epochs
        self.seed = seed
        self.scaler = Standardizer()
        self.params = None

    def _forward(self, X):
        h = np.maximum(0.0, X @ self.params["W1"] + self.params["b1"])  # ReLU
        return h, sigmoid(h @ self.params["W2"] + self.params["b2"])

    def fit(self, X, y):
        X = self.scaler.fit_transform(to_array(X))
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / d), (d, self.hidden)),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, np.sqrt(2 / self.hidden), (self.hidden, 1)),
            "b2": np.zeros(1),
        }
        for _ in range(self.epochs):
            h, p = self._forward(X)
            dz2 = (p - y) / n
            dh = (dz2 @ self.params["W2"].T) * (h > 0)
            grads = {
                "W2": h.T @ dz2, "b2": dz2.sum(axis=0),
                "W1": X.T @ dh, "b1": dh.sum(axis=0),
            }
            self.optimizer.update(self.params, grads)
        return self

    def predict(self, X):
        if self.params is None:
            raise RuntimeError("Le modèle n'est pas entraîné : appelez fit() d'abord.")
        X = self.scaler.transform(to_array(X))
        return self._forward(X)[1].ravel()
