# tests/test_ml_models/test_evaluator.py

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.ml_models.evaluator import evaluate_classification_model
from src.ml_models.model_builder import ModelBuilder


@pytest.fixture
def synthetic_data():
    X = np.array([[0.], [1.], [0.], [1.]])
    y = np.array([0, 1, 0, 1])
    model = ModelBuilder.build_logistic_regression(pca_components=None)
    model.fit(X, y)
    return X, y, model


def test_evaluate_default_metrics(synthetic_data):
    X, y, model = synthetic_data
    results = evaluate_classification_model(model, X, y)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        assert metric in results
        assert 0.0 <= results[metric] <= 1.0


def test_evaluate_custom_metrics_and_save(tmp_path, synthetic_data):
    X, y, model = synthetic_data
    out = tmp_path / "report"
    results = evaluate_classification_model(
        model,
        X,
        y,
        metrics=['accuracy', 'f1'],
        output_path=str(out)
    )
    assert set(results.keys()) == {'accuracy', 'f1'}
    # Check files created
    assert os.path.exists(str(out) + '.csv')
    assert os.path.exists(str(out) + '.json')
    df = pd.read_csv(str(out) + '.csv')
    assert 'accuracy' in df.columns and 'f1' in df.columns
    with open(str(out) + '.json') as f:
        data = json.load(f)
    assert 'accuracy' in data and 'f1' in data
