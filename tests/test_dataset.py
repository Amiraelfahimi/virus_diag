from virus_diag.core.dataset import PatientDataset
from virus_diag.utils.preprocessing import FEATURES


def test_dataset_loads():
    ds = PatientDataset()
    assert len(ds) == 500
    assert list(ds.X.columns) == FEATURES


def test_split_sizes_and_no_overlap():
    ds = PatientDataset()
    X_train, X_test, y_train, y_test = ds.split(test_size=0.2, seed=42)
    assert len(X_train) + len(X_test) == len(ds)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_is_stratified():
    ds = PatientDataset()
    _, _, y_train, y_test = ds.split()
    assert abs(y_train.mean() - y_test.mean()) < 0.05
