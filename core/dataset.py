"""Chargement et découpage du jeu de données patients."""
from pathlib import Path

import numpy as np
import pandas as pd

from virus_diag.utils.preprocessing import FEATURES, clean

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "patient_data.csv"


class PatientDataset:
    def __init__(self, path=DEFAULT_PATH, features=FEATURES, target="infected"):
        self.path = Path(path)
        self.features = list(features)
        self.target = target
        self.data = clean(pd.read_csv(self.path), self.features, self.target)

    def __len__(self):
        return len(self.data)

    @property
    def X(self):
        return self.data[self.features]

    @property
    def y(self):
        return self.data[self.target]

    def summary(self):
        """Résumé du jeu de données (taille, classes, statistiques)."""
        return {
            "n_patients": len(self),
            "taux_infectes": float(self.y.mean()),
            "statistiques": self.X.describe().T[["mean", "std", "min", "max"]].round(2),
        }

    def split(self, test_size=0.2, seed=42):
        """Découpe train/test en conservant la proportion de chaque classe."""
        rng = np.random.default_rng(seed)
        test_idx = []
        for label in self.y.unique():
            idx = np.flatnonzero(self.y.to_numpy() == label)
            rng.shuffle(idx)
            test_idx.extend(idx[: int(round(len(idx) * test_size))])
        mask = np.zeros(len(self), dtype=bool)
        mask[test_idx] = True
        return self.X[~mask], self.X[mask], self.y[~mask], self.y[mask]
