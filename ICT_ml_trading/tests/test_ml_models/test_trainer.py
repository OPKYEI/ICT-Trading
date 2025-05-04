# tests/test_ml_models/test_trainer.py
import pytest
import numpy as np
from src.ml_models.trainer import train_with_grid_search, train_with_random_search
from src.ml_models.model_builder import ModelBuilder

@ pytest.fixture
def synthetic_data():
    # Very small synthetic binary classification data
    X = np.array([[0.1], [1.2], [0.2], [1.1], [0.3], [1.3]])
    y = np.array([0, 1, 0, 1, 0, 1])
    return X, y


def test_train_with_grid_search_returns_estimator_and_results(synthetic_data):
    X, y = synthetic_data
    # Simple logistic regression pipeline
    model = ModelBuilder.build_logistic_regression(pca_components=None)
    # Grid over regularization
    param_grid = {'clf__C': [0.01, 1.0]}
    best_model, results = train_with_grid_search(
        model,
        param_grid,
        X,
        y,
        cv=2,
        scoring='accuracy'
    )
    # Best model can predict
    preds = best_model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert 'mean_test_score' in results
    assert results['mean_test_score'].shape[0] == len(param_grid['clf__C'])


def test_train_with_random_search_returns_estimator_and_results(synthetic_data):
    X, y = synthetic_data
    # Random forest pipeline
    model = ModelBuilder.build_random_forest(n_estimators=5, pca_components=None)
    param_dists = {'clf__n_estimators': [2, 5]}
    best_model, results = train_with_random_search(
        model,
        param_dists,
        X,
        y,
        cv=2,
        n_iter=2,
        random_state=42
    )
    # Best model can predict
    preds = best_model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert 'mean_test_score' in results
    assert len(results['mean_test_score']) == 2
