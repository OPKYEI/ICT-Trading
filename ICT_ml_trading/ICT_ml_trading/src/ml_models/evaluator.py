# src/ml_models/evaluator.py

import json
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def evaluate_classification_model(
    model: Any,
    X: Any,
    y_true: Any,
    metrics: Optional[list[str]] = None,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a fitted classification model.

    Args:
        model: A fitted sklearn-like estimator with predict() and predict_proba() or decision_function().
        X: Features for evaluation.
        y_true: True binary labels.
        metrics: List of metrics to compute (default all).
        output_path: If given, saves results to {output_path}.csv and {output_path}.json.

    Returns:
        Dictionary mapping metric names to their values.
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    y_pred = model.predict(X)
    results: Dict[str, float] = {}

    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    if 'precision' in metrics:
        results['precision'] = precision_score(y_true, y_pred)
    if 'recall' in metrics:
        results['recall'] = recall_score(y_true, y_pred)
    if 'f1' in metrics:
        results['f1'] = f1_score(y_true, y_pred)
    if 'roc_auc' in metrics:
        try:
            y_score = model.predict_proba(X)[:, 1]
        except AttributeError:
            y_score = model.decision_function(X)
        results['roc_auc'] = roc_auc_score(y_true, y_score)

    if output_path:
        df = pd.DataFrame([results])
        df.to_csv(output_path + '.csv', index=False)
        with open(output_path + '.json', 'w') as f:
            json.dump(results, f)

    return results
