"""Interface commune à tous les modèles d'IA du projet."""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Contrat : fit() entraîne, predict() renvoie une probabilité d'infection.

    Tout modèle qui respecte ce contrat peut être utilisé par ClinicalPredictor
    et par le Trainer / Evaluator sans qu'ils aient besoin de le connaître.
    """

    @abstractmethod
    def fit(self, X, y):
        """Entraîne le modèle et retourne self."""

    @abstractmethod
    def predict(self, X):
        """Retourne la probabilité (0 à 1) d'être infecté, une par patient."""

    def predict_label(self, X, threshold=0.5):
        """Classe 0/1 selon le seuil."""
        import numpy as np
        return (np.asarray(self.predict(X)).ravel() >= threshold).astype(int)
