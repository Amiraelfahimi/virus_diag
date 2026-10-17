"""Métriques d'évaluation pour la classification binaire."""
import numpy as np


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(a, b):
    return a / b if b else 0.0


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred):
    c = confusion_matrix(y_true, y_pred)
    return _safe_div(c["tp"], c["tp"] + c["fp"])


def recall(y_true, y_pred):
    c = confusion_matrix(y_true, y_pred)
    return _safe_div(c["tp"], c["tp"] + c["fn"])


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _safe_div(2 * p * r, p + r)


def binary_cross_entropy(y_true, y_proba, eps=1e-12):
    p = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
