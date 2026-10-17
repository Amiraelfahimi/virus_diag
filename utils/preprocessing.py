"""Prétraitement des données patient."""
import numpy as np
import pandas as pd

FEATURES = [
    "age", "temperature_c", "cough", "fatigue",
    "oxygen_saturation", "white_blood_cells",
]


def clean(df, features=FEATURES, target="infected"):
    """Vérifie les colonnes et supprime les lignes avec valeurs manquantes."""
    missing = [c for c in [*features, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes du fichier : {missing}")
    return df.dropna(subset=[*features, target]).reset_index(drop=True)


def to_array(patient_data, features=FEATURES):
    """Convertit un DataFrame, une Series, un dict ou un tableau en matrice 2D."""
    try:
        if isinstance(patient_data, pd.DataFrame):
            return patient_data[features].to_numpy(dtype=float)
        if isinstance(patient_data, pd.Series):
            return patient_data[features].to_numpy(dtype=float).reshape(1, -1)
        if isinstance(patient_data, dict):
            return np.array([[patient_data[f] for f in features]], dtype=float)
    except KeyError as e:
        raise ValueError(f"Variable patient manquante : {e}") from None
    return np.atleast_2d(np.asarray(patient_data, dtype=float))


class Standardizer:
    """Centre et réduit les variables (moyenne 0, écart-type 1).

    fit() apprend moyenne/écart-type sur le jeu d'entraînement UNIQUEMENT ;
    transform() les réutilise sur tout nouveau patient (évite la fuite de données).
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Standardizer non ajusté : appelez fit() d'abord.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
