"""Point d'entrée de virus_diag.

Depuis le dossier virus_diag/ :
    python main.py train                      # entraîne + évalue (régression logistique)
    python main.py train --model neural       # idem avec le réseau de neurones
    python main.py predict --age 65 --temperature_c 38.9 --cough 1 --fatigue 0 \
                           --oxygen_saturation 93.5 --white_blood_cells 11.2
    python main.py data                       # inspecte dataset + prétraitement
    python main.py demo                       # interface clinique en console
    python main.py api --port 8000            # API HTTP
"""
import argparse
import sys
from pathlib import Path

# Permet de lancer "python main.py" depuis virus_diag/ (ajoute le dossier parent au path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virus_diag.core.clinical_predictor import ClinicalPredictor
from virus_diag.core.logistic_regression import LogisticRegressionModel
from virus_diag.core.neural_network import NeuralNetworkModel
from virus_diag.pipeline.evaluator import Evaluator
from virus_diag.pipeline.trainer import Trainer
from virus_diag.utils.preprocessing import FEATURES

MODELS = {"logistic": LogisticRegressionModel, "neural": NeuralNetworkModel}


def train_and_evaluate(model_name):
    trainer = Trainer(MODELS[model_name]())
    model = trainer.train()
    scores = Evaluator(model).evaluate(trainer.X_test, trainer.y_test)
    print(f"Modèle : {model_name}")
    for k in ("accuracy", "precision", "recall", "f1", "loss"):
        print(f"  {k:<10}: {scores[k]:.3f}")
    print(f"  confusion : {scores['confusion']}")
    return model


def main():
    parser = argparse.ArgumentParser(description="virus_diag - diagnostic pédagogique")
    sub = parser.add_subparsers(dest="cmd")

    for name in ("train", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=MODELS, default="logistic")
    pred = sub.choices["predict"]
    for f in FEATURES:
        pred.add_argument(f"--{f}", type=float, required=True)
    sub.add_parser("data")
    sub.add_parser("demo")
    api = sub.add_parser("api")
    api.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.cmd in (None, "train"):
        train_and_evaluate(getattr(args, "model", "logistic"))
    elif args.cmd == "predict":
        model = Trainer(MODELS[args.model]()).train()
        patient = {f: getattr(args, f) for f in FEATURES}
        print("Diagnostic :", ClinicalPredictor(model).diagnose(patient))
    elif args.cmd == "data":
        from virus_diag.core.dataset import PatientDataset
        from virus_diag.utils.preprocessing import Standardizer, to_array
        ds = PatientDataset()
        info = ds.summary()
        print(f"Patients : {info['n_patients']} | taux d'infectés : {info['taux_infectes']:.1%}")
        print(info["statistiques"].to_string())
        X_train, X_test, y_train, y_test = ds.split()
        print(f"\nSplit stratifié : train={len(X_train)}, test={len(X_test)}")
        Z = Standardizer().fit_transform(to_array(X_train))
        print("Après standardisation (train) : moyenne ≈", Z.mean(axis=0).round(2),
              "| écart-type ≈", Z.std(axis=0).round(2))
    elif args.cmd == "demo":
        from virus_diag.app.interface_clinique import main as demo
        demo()
    elif args.cmd == "api":
        from virus_diag.app.api import serve
        serve(args.port)


if __name__ == "__main__":
    main()
