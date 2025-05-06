# src/ml_models/trainer.py

import os
import pickle
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List

from sklearn.model_selection import ParameterGrid, train_test_split, cross_val_score
from sklearn.base import BaseEstimator, clone
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


def grid_search_with_checkpoint(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'accuracy',
    checkpoint_path: str = 'grid_checkpoint.pkl',
    resume: bool = True
) -> Tuple[BaseEstimator, pd.DataFrame]:
    """
    Performs grid search with checkpoint support and final training on full data.
    """
    grid = list(ParameterGrid(param_grid))
    results = []
    start = 0

    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            saved = pickle.load(f)
            results = saved['results']
            start = len(results)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for params in tqdm(grid[start:], total=len(grid), desc=f"GridSearch {model.__class__.__name__}"):
        clf = clone(model).set_params(**params)
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append({**params, 'mean_test_score': scores.mean()})
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'results': results}, f)

    df = pd.DataFrame(results)
    best_row = df.loc[df['mean_test_score'].idxmax()].to_dict()
    best_params = {k: best_row[k] for k in param_grid}

    # Cast float → int where needed
    for k, v in best_params.items():
        if isinstance(v, float) and v.is_integer():
            best_params[k] = int(v)

    best_model = clone(model).set_params(**best_params).fit(X, y)
    return best_model, df


def train_multiple_models_with_split(
    models: Dict[str, Tuple[BaseEstimator, Dict[str, Any]]],
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_prefix: str = "model",
    scoring: str = 'accuracy'
) -> Dict[str, Dict[str, Any]]:
    """
    Trains multiple models with 80/20 split and logs results.
    Returns best model objects and performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    results = {}

    for name, (model, param_grid) in models.items():
        print(f"\n🔍 Training {name}...")
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_{name}_checkpoint.pkl")
        trained_model, df_results = grid_search_with_checkpoint(
            model=model,
            param_grid=param_grid,
            X=X_train,
            y=y_train,
            cv=3,
            scoring=scoring,
            checkpoint_path=checkpoint_path,
            resume=True
        )

        # Final evaluation on holdout set
        y_pred = trained_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"✅ {name} accuracy on holdout test: {acc:.4f}")
        results[name] = {
            "model": trained_model,
            "grid_results": df_results,
            "test_accuracy": acc
        }

    return results
