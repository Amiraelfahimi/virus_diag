import numpy as np
import pandas as pd
import pytest

from virus_diag.utils.preprocessing import FEATURES, Standardizer, clean, to_array


def test_clean_drops_missing_values():
    df = pd.DataFrame({**{f: [1.0, None] for f in FEATURES}, "infected": [0, 1]})
    assert len(clean(df)) == 1


def test_clean_raises_on_missing_column():
    with pytest.raises(ValueError):
        clean(pd.DataFrame({"age": [1]}))


def test_to_array_from_dict():
    patient = {f: 1.0 for f in FEATURES}
    assert to_array(patient).shape == (1, len(FEATURES))


def test_to_array_missing_key():
    with pytest.raises(ValueError):
        to_array({"age": 30})


def test_standardizer_mean_zero_std_one():
    X = np.random.default_rng(0).normal(5, 3, size=(100, 3))
    Z = Standardizer().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-9)


def test_standardizer_requires_fit():
    with pytest.raises(RuntimeError):
        Standardizer().transform(np.zeros((1, 2)))
