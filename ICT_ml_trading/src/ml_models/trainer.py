# src/ml_models/trainer.py

import os
import pickle
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import ParameterGrid, ParameterSampler, cross_val_score
from tqdm.auto import tqdm


def grid_search_with_checkpoint(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str = 'accuracy',
    checkpoint_path: str = 'grid_checkpoint.pkl',
    resume: bool = True
) -> Tuple[BaseEstimator, pd.DataFrame]:
    grid = list(ParameterGrid(param_grid))
    results = []
    start = 0

    if resume and os.path.exists(checkpoint_path):
        saved = pickle.load(open(checkpoint_path, 'rb'))
        results = saved['results']
        start = len(results)

    for params in tqdm(grid[start:], total=len(grid), desc='GridSearch'):
        clf = clone(model).set_params(**params)
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append({**params, 'mean_test_score': scores.mean()})
        pickle.dump({'results': results}, open(checkpoint_path, 'wb'))

    df = pd.DataFrame(results)
    best_row = df.loc[df['mean_test_score'].idxmax()].to_dict()
    best_params = {k: best_row[k] for k in param_grid}

    # Cast float ints back to int for tree-based params
    for k, v in best_params.items():
        if isinstance(v, float) and v.is_integer():
            best_params[k] = int(v)

    best_model = clone(model).set_params(**best_params).fit(X, y)
    return best_model, df


def random_search_with_checkpoint(
    model: BaseEstimator,
    param_distributions: Dict[str, Any],
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_iter: int = 20,
    random_state: Optional[int] = None,
    checkpoint_path: str = 'random_checkpoint.pkl',
    resume: bool = True
) -> Tuple[BaseEstimator, pd.DataFrame]:
    sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
    results = []
    start = 0

    if resume and os.path.exists(checkpoint_path):
        saved = pickle.load(open(checkpoint_path, 'rb'))
        results = saved['results']
        start = len(results)

    for params in tqdm(sampler[start:], total=len(sampler), desc='RandomSearch'):
        clf = clone(model).set_params(**params)
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append({**params, 'mean_test_score': scores.mean()})
        pickle.dump({'results': results}, open(checkpoint_path, 'wb'))

    df = pd.DataFrame(results)
    best_row = df.loc[df['mean_test_score'].idxmax()].to_dict()
    best_params = {k: best_row[k] for k in param_distributions}

    # Cast float ints back to int for tree-based params
    for k, v in best_params.items():
        if isinstance(v, float) and v.is_integer():
            best_params[k] = int(v)

    best_model = clone(model).set_params(**best_params).fit(X, y)
    return best_model, df
