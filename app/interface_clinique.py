"""Interface clinique en ligne de commande (démo pédagogique).

Lancement : python -m virus_diag.app.interface_clinique
"""
from virus_diag.core.clinical_predictor import ClinicalPredictor
from virus_diag.pipeline.evaluator import Evaluator
from virus_diag.pipeline.trainer import Trainer
from virus_diag.utils.preprocessing import FEATURES


def saisir_patient():
    print("Saisissez les données du patient :")
    return {f: float(input(f"  {f} : ")) for f in FEATURES}


def main():
    trainer = Trainer()
    model = trainer.train()
    scores = Evaluator(model).evaluate(trainer.X_test, trainer.y_test)
    print(f"Modèle entraîné - accuracy de test : {scores['accuracy']:.3f}")
    print("(Données synthétiques : ne pas utiliser pour un vrai diagnostic.)\n")

    predictor = ClinicalPredictor(model)
    exemple = trainer.X_test.iloc[0]
    print("Patient exemple :", exemple.to_dict())
    print("Diagnostic     :", predictor.diagnose(exemple), "\n")

    while input("Tester un autre patient ? (o/N) ").strip().lower() == "o":
        print("Diagnostic :", predictor.diagnose(saisir_patient()), "\n")


if __name__ == "__main__":
    main()
