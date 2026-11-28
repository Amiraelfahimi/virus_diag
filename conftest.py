"""Rend le package `virus_diag` importable pendant les tests.

Le dossier racine du dépôt EST le package : on ajoute son dossier parent au path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
