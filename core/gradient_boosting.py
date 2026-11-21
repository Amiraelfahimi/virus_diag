"""Gradient Boosting (scikit-learn) adapté à l’interface BaseModel du projet."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from virus_diag.core.model import BaseModel
from virus_diag.utils.preprocessing import to_array


class GradientBoostingModel(BaseModel):
    def __init__(
        self,
        max_iter=200,
        learning_rate=0.1,
        max_depth=None,
        random_state=42,
    ):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )),
        ])

    def fit(self, X, y):
        """Entraîne le modèle et retourne self."""
        X = to_array(X)
        y = np.asarray(y).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Retourne la probabilité d’être infecté (classe 1)."""
        X = to_array(X)
        return self.model.predict_proba(X)[:, 1]