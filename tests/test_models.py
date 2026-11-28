import numpy as np
import pytest

from virus_diag.core.clinical_predictor import ClinicalPredictor
from virus_diag.core.gradient_boosting import GradientBoostingModel
from virus_diag.core.logistic_regression import LogisticRegressionModel
from virus_diag.core.neural_network import NeuralNetworkModel
from virus_diag.core.RandomForestModel import RandomForestModel
from virus_diag.pipeline.evaluator import Evaluator
from virus_diag.pipeline.trainer import Trainer

MODEL_CLASSES = [
    LogisticRegressionModel,
    NeuralNetworkModel,
    RandomForestModel,
    GradientBoostingModel,
]


class FakeModel:
    def __init__(self, proba):
        self.proba = proba

    def predict(self, X):
        return np.array([self.proba])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LogisticRegressionModel().predict(np.zeros((1, 6)))


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_trainer_and_evaluator(model_cls):
    trainer = Trainer(model_cls())
    model = trainer.train()
    proba = model.predict(trainer.X_test)
    assert len(proba) == len(trainer.X_test)
    assert ((proba >= 0) & (proba <= 1)).all()
    scores = Evaluator(model).evaluate(trainer.X_test, trainer.y_test)
    assert scores["accuracy"] > 0.6


def test_clinical_predictor_threshold():
    assert ClinicalPredictor(FakeModel(0.9)).diagnose({}) == "Infecté"
    assert ClinicalPredictor(FakeModel(0.5)).diagnose({}) == "Infecté"
    assert ClinicalPredictor(FakeModel(0.1)).diagnose({}) == "Sain"


def test_clinical_predictor_rejects_object_without_predict():
    with pytest.raises(TypeError):
        ClinicalPredictor(object())
