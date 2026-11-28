"""Trainer : seule classe responsable de l'entraînement."""
from virus_diag.core.dataset import PatientDataset
from virus_diag.core.logistic_regression import LogisticRegressionModel
from virus_diag.core.random_forest import RandomForestModel
from virus_diag.core.gradient_boosting import GradientBoostingModel


class Trainer:
    def __init__(self, model=None, dataset=None, test_size=0.2, seed=42):
        # Par défaut on garde la régression logistique
        self.model = model or LogisticRegressionModel()
        self.dataset = dataset or PatientDataset()
        self.test_size = test_size
        self.seed = seed
        self.X_test = None
        self.y_test = None

    def train(self):
        """Entraîne le modèle et retourne le modèle entraîné."""
        X_train, self.X_test, y_train, self.y_test = self.dataset.split(
            self.test_size, self.seed
        )
        self.model.fit(X_train, y_train)
        return self.model


if __name__ == "__main__":
    from virus_diag.pipeline.evaluator import Evaluator

    # Exemples d'utilisation :
    # trainer = Trainer()                                    # Logistic (défaut)
    # trainer = Trainer(model=RandomForestModel())           # Random Forest
    trainer = Trainer(model=GradientBoostingModel())         # Gradient Boosting

    trained = trainer.train()
    print(Evaluator(trained).evaluate(trainer.X_test, trainer.y_test))