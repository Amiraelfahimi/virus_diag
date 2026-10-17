"""ClinicalPredictor : utilise un modèle déjà entraîné pour poser un diagnostic.

Il ne connaît ni l'entraînement, ni l'algorithme : il exige seulement que le
modèle possède une méthode predict() renvoyant une probabilité (duck typing /
polymorphisme).
"""
import numpy as np


class ClinicalPredictor:
    THRESHOLD = 0.5

    def __init__(self, trained_model):
        if not callable(getattr(trained_model, "predict", None)):
            raise TypeError("Le modèle doit avoir une méthode predict().")
        self.model = trained_model

    def diagnose(self, patient_data):
        """Retourne "Infecté" si prédiction >= 0.5, "Sain" sinon."""
        proba = np.asarray(self.model.predict(patient_data), dtype=float).ravel()
        if proba.size != 1:
            raise ValueError("diagnose() attend les données d'un seul patient.")
        return "Infecté" if proba[0] >= self.THRESHOLD else "Sain"
