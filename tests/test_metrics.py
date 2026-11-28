import pytest

from virus_diag.utils import metrics


def test_confusion_matrix():
    c = metrics.confusion_matrix([1, 1, 0, 0], [1, 0, 0, 1])
    assert c == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_perfect_prediction():
    y = [0, 1, 1, 0]
    assert metrics.accuracy(y, y) == 1.0
    assert metrics.precision(y, y) == 1.0
    assert metrics.recall(y, y) == 1.0
    assert metrics.f1_score(y, y) == 1.0


def test_no_positive_prediction_gives_zero_not_error():
    assert metrics.precision([1, 0], [0, 0]) == 0.0
    assert metrics.f1_score([1, 0], [0, 0]) == 0.0


def test_binary_cross_entropy_small_for_good_probabilities():
    assert metrics.binary_cross_entropy([1, 0], [0.99, 0.01]) < 0.05
    assert metrics.binary_cross_entropy([1, 0], [0.01, 0.99]) > 2
    assert metrics.accuracy([1, 0], [1, 0]) == pytest.approx(1.0)
