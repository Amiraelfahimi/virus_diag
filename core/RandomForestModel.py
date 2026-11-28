"""Random Forest (scikit-learn) adapté à l’interface BaseModel du projet."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from virus_diag.core.model import BaseModel
from virus_diag.utils.preprocessing import to_array


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=200, max_depth=None, random_state=42, n_jobs=-1):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs,
            )),
        ])

    def fit(self, X, y):
        """Entraîne le modèle et retourne self (contrat BaseModel)."""
        X = to_array(X)
        y = np.asarray(y).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Retourne la probabilité d’être infecté (classe 1), une par patient."""
        X = to_array(X)
        # predict_proba renvoie [proba_classe_0, proba_classe_1]
        return self.model.predict_proba(X)[:, 1]