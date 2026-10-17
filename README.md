# virus_diag — Chap. 5 : POO adaptée à l'IA

Données synthétiques : usage pédagogique uniquement, pas de vrai diagnostic.

## Structure
```
virus_diag/
├── data/patient_data.csv
├── core/      dataset.py, model.py, logistic_regression.py, neural_network.py,
│              optimizer.py, clinical_predictor.py
├── pipeline/  trainer.py, evaluator.py
├── utils/     preprocessing.py, metrics.py
└── app/       api.py, interface_clinique.py
```

## Lancer (depuis le dossier `virus_diag/`)
```
python main.py train                  # entraîne + évalue
python main.py train --model neural   # avec le réseau de neurones
python main.py predict --age 65 --temperature_c 38.9 --cough 1 --fatigue 0 --oxygen_saturation 93.5 --white_blood_cells 11.2
python main.py data                   # inspecte dataset + prétraitement
python main.py demo                   # interface clinique (console)
python main.py api --port 8000        # API POST /diagnose
```
Dépendances : `numpy`, `pandas`.

## Réponses

**1. ClinicalPredictor** : voir `core/clinical_predictor.py`.
Il reçoit un modèle déjà entraîné, `diagnose(patient_data)` renvoie
"Infecté" si la prédiction est ≥ 0.5, sinon "Sain".

**2. Concept POO** : le **polymorphisme**, ici sous forme de *duck typing*
(et d'interface avec `BaseModel`). `ClinicalPredictor` ne dépend pas d'une
classe précise mais d'un contrat : « l'objet a une méthode `predict()` ».
`LogisticRegressionModel`, `NeuralNetworkModel` ou tout futur modèle peuvent
être injectés (injection de dépendance) sans modifier `ClinicalPredictor`.

**3. Pourquoi séparer prédiction et entraînement** (principe de responsabilité
unique / séparation des préoccupations) :
- Ils n'ont pas le même cycle de vie : l'entraînement est lourd, rare et
  hors-ligne ; la prédiction est légère, fréquente et en production.
- Le code clinique (API, interface) n'a pas besoin des données
  d'entraînement ni de l'optimiseur : moins de dépendances, moins de risques
  (fuite de données, ré-entraînement accidentel).
- On peut changer ou ré-entraîner le modèle sans toucher à la logique de
  diagnostic, et inversement (changer le seuil, le format de sortie).
- Les deux se testent séparément (un faux modèle suffit pour tester
  `ClinicalPredictor`) ; l'évaluation reste reproductible.
