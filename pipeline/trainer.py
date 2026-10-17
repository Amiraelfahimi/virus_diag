"""Trainer : seule classe responsable de l'entraînement."""
from virus_diag.core.dataset import PatientDataset
from virus_diag.core.logistic_regression import LogisticRegressionModel


class Trainer:
    def __init__(self, model=None, dataset=None, test_size=0.2, seed=42):
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

    trainer = Trainer()
    trained = trainer.train()
    print(Evaluator(trained).evaluate(trainer.X_test, trainer.y_test))
