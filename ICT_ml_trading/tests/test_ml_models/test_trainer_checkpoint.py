# tests/test_ml_models/test_trainer_checkpoint.py
import os
import pickle
import pytest
import numpy as np
import pandas as pd
from src.ml_models.trainer import grid_search_with_checkpoint, random_search_with_checkpoint
from src.ml_models.model_builder import ModelBuilder

@ pytest.fixture
def synthetic_data(tmp_path):
    # Create tiny binary classification data
    X = np.array([[0.1], [1.0], [0.2], [1.1], [0.3], [1.2]])
    y = np.array([0, 1, 0, 1, 0, 1])
    # Create checkpoint paths
    grid_chk = tmp_path / "grid_checkpoint.pkl"
    rnd_chk = tmp_path / "random_checkpoint.pkl"
    return X, y, str(grid_chk), str(rnd_chk)


def test_grid_search_with_checkpoint_creates_file_and_resumes(synthetic_data):
    X, y, grid_chk, _ = synthetic_data
    # Ensure no preexisting file
    if os.path.exists(grid_chk):
        os.remove(grid_chk)
    model = ModelBuilder.build_logistic_regression(pca_components=None)
    param_grid = {'clf__C': [0.01, 1.0]}

    # First run without resume: should create checkpoint
    best_model1, df1 = grid_search_with_checkpoint(
        model, param_grid, X, y, cv=2, scoring='accuracy', checkpoint_path=grid_chk, resume=False
    )
    assert os.path.exists(grid_chk)
    assert isinstance(df1, pd.DataFrame)
    assert df1.shape[0] == len(param_grid['clf__C'])
    preds1 = best_model1.predict(X)
    assert isinstance(preds1, np.ndarray)

    # Modify checkpoint to simulate partial run
    with open(grid_chk, 'rb') as f:
        saved = pickle.load(f)
    # Drop last result to simulate incomplete
    partial = saved['results'][:-1]
    pickle.dump({'results': partial}, open(grid_chk, 'wb'))

    # Resume should process only missing entry
    best_model2, df2 = grid_search_with_checkpoint(
        model, param_grid, X, y, cv=2, scoring='accuracy', checkpoint_path=grid_chk, resume=True
    )
    assert df2.shape[0] == len(param_grid['clf__C'])
    preds2 = best_model2.predict(X)
    assert isinstance(preds2, np.ndarray)


def test_random_search_with_checkpoint_creates_file_and_resumes(synthetic_data):
    X, y, _, rnd_chk = synthetic_data
    if os.path.exists(rnd_chk):
        os.remove(rnd_chk)
    model = ModelBuilder.build_random_forest(n_estimators=5, pca_components=None)
    param_dist = {'clf__n_estimators': [2, 5]}

    # First run
    best_model1, df1 = random_search_with_checkpoint(
        model, param_dist, X, y, cv=2, scoring='accuracy', n_iter=2,
        checkpoint_path=rnd_chk, resume=False
    )
    assert os.path.exists(rnd_chk)
    assert isinstance(df1, pd.DataFrame)
    assert df1.shape[0] == 2
    preds1 = best_model1.predict(X)
    assert isinstance(preds1, np.ndarray)

    # Partial simulate
    with open(rnd_chk, 'rb') as f:
        saved = pickle.load(f)
    partial = saved['results'][:1]
    pickle.dump({'results': partial}, open(rnd_chk, 'wb'))

    # Resume
    best_model2, df2 = random_search_with_checkpoint(
        model, param_dist, X, y, cv=2, scoring='accuracy', n_iter=2,
        checkpoint_path=rnd_chk, resume=True
    )
    assert df2.shape[0] == 2
    preds2 = best_model2.predict(X)
    assert isinstance(preds2, np.ndarray)
