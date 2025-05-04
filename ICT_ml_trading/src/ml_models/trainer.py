# src/ml_models/trainer.py
"""
Training and hyperparameter search scaffolding for ML pipelines.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator

def train_with_grid_search(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_jobs: int = -1,
    refit: bool = True
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Perform GridSearchCV on a given estimator pipeline.

    Args:
        model: Pipeline or estimator implementing .fit()
        param_grid: dict of hyperparameters for grid search
        X: feature data
        y: target labels
        cv: number of cross-validation folds
        scoring: metric name for scoring
        n_jobs: parallel jobs
        refit: whether to refit best model on full data

    Returns:
        best_model: estimator refit on full data
        results: dictionary of cv results
    """
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        return_train_score=True
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.cv_results_


def train_with_random_search(
    model: BaseEstimator,
    param_distributions: Dict[str, Any],
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_iter: int = 20,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    refit: bool = True
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Perform RandomizedSearchCV on a given estimator pipeline.

    Args:
        model: Pipeline or estimator implementing .fit()
        param_distributions: dict of hyperparameter distributions
        X: feature data
        y: target labels
        cv: number of cross-validation folds
        scoring: metric name for scoring
        n_iter: number of parameter settings sampled
        n_jobs: parallel jobs
        random_state: seed for reproducibility
        refit: whether to refit best model on full data

    Returns:
        best_model: estimator refit on full data
        results: dictionary of cv results
    """
    rnd = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=refit,
        return_train_score=True
    )
    rnd.fit(X, y)
    return rnd.best_estimator_, rnd.cv_results_
