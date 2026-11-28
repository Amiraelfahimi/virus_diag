from virus_diag.app.api import DiagnosisAPI
from virus_diag.core.clinical_predictor import ClinicalPredictor
from virus_diag.pipeline.trainer import Trainer
from virus_diag.utils.preprocessing import FEATURES


def _api():
    return DiagnosisAPI(ClinicalPredictor(Trainer().train()))


def test_api_missing_fields():
    status, body = _api().handle({"age": 40})
    assert status == 400
    assert "erreur" in body


def test_api_diagnose_ok():
    patient = dict(zip(FEATURES, [65, 38.9, 1, 0, 93.5, 11.2]))
    status, body = _api().handle(patient)
    assert status == 200
    assert body["diagnostic"] in ("Infecté", "Sain")
