# src/ml_models/trainer.py

import os
import pickle
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from sklearn.model_selection import ParameterGrid, train_test_split, cross_val_score
from sklearn.base import BaseEstimator, clone
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import get_scorer
       

def grid_search_with_checkpoint(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = 'accuracy',
    checkpoint_dir: str = 'checkpoints',
    prefix: str = 'grid'
) -> Tuple[BaseEstimator, float, List[float]]:
    """
    📊 Nested TimeSeriesSplit CV:
    1️⃣ Outer loop for honest test performance (outer_splits)
    2️⃣ Inner loop for hyperparam tuning (inner_splits)
    Returns final_model trained on full data, average outer score, and per-fold scores.
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    outer_cv = TimeSeriesSplit(n_splits=outer_splits)
    inner_cv = TimeSeriesSplit(n_splits=inner_splits)
    scorer = get_scorer(scoring)

    outer_scores: List[float] = []
    best_params_list: List[Dict[str, Any]] = []

    # Outer folds
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"📈 Outer fold {fold}/{outer_splits}")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        # Inner grid search
        best_score = float('-inf')
        best_params: Dict[str,Any] = {}
        for params in ParameterGrid(param_grid):
            clf = clone(model).set_params(**params)
            scores = []
            for _, (itr, ivl) in enumerate(inner_cv.split(X_tr)):
                X_itr, y_itr = X_tr.iloc[itr], y_tr.iloc[itr]
                X_ivl, y_ivl = X_tr.iloc[ivl], y_tr.iloc[ivl]
                clf.fit(X_itr, y_itr)
                scores.append(scorer(clf, X_ivl, y_ivl))
            mean_inner = sum(scores) / len(scores)
            if mean_inner > best_score:
                best_score = mean_inner
                best_params = params

        print(f"🔍 Best inner params fold {fold}: {best_params}, score {best_score:.4f}")
        best_params_list.append(best_params)

        # Retrain on full train split
        best_clf = clone(model).set_params(**best_params).fit(X_tr, y_tr)
        outer_score = scorer(best_clf, X_te, y_te)
        print(f"✅ Outer fold {fold} test {scoring}: {outer_score:.4f}")
        outer_scores.append(outer_score)

        # Checkpoint each fold's model
        ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_{model.__class__.__name__}_fold{fold}.pkl")
        with open(ckpt_path, 'wb') as f:
            pickle.dump({'params': best_params}, f)

    avg_score = sum(outer_scores) / len(outer_scores)
    print(f"📊 Average nested CV {scoring}: {avg_score:.4f}")

    # Choose most common best_params
    from collections import Counter
    param_tuples = [tuple(sorted(p.items())) for p in best_params_list]
    most_common = Counter(param_tuples).most_common(1)[0][0]
    final_params = dict(most_common)

    # Final model on full data
    final_model = clone(model).set_params(**final_params).fit(X, y)
    return final_model, avg_score, outer_scores

def train_multiple_models_with_split(
    models: Dict[str, Tuple[BaseEstimator, Dict[str, Any]]],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_prefix: str = "model",
    scoring: str = 'accuracy'
) -> Dict[str, float]:
    """
    📈 Walk-forward (rolling) analysis:
      - Split data into n_splits successive folds
      - In each fold: grid-search on train window, test on next window
      - Report average accuracy per model
    """
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare rolling splitter
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results: Dict[str, List[float]] = {name: [] for name in models}

    # Loop over each fold
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n📈 Walk-forward fold {fold}/{n_splits}")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[test_idx],  y.iloc[test_idx]

        # For each model, tune then test
        for name, (model, param_grid) in models.items():
            print(f"🔍 Fold {fold} training {name}...")
            # Checkpoint per fold
            ckpt = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_{name}_fold{fold}.pkl")
            best_model, _ = grid_search_with_checkpoint(
                model=model,
                param_grid=param_grid,
                X=X_tr,
                y=y_tr,
                cv=TimeSeriesSplit(n_splits=3),
                scoring=scoring,
                checkpoint_path=ckpt,
                resume=True
            )
            # Evaluate on validation slice
            y_pred = best_model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"✅ Fold {fold} {name} accuracy: {acc:.4f}")
            fold_results[name].append(acc)

    # Compute & return average performance
    avg_results: Dict[str, float] = {}
    for name, scores in fold_results.items():
        avg = sum(scores) / len(scores)
        print(f"\n📊 {name} average walk-forward accuracy: {avg:.4f}")
        avg_results[name] = avg

    return avg_results
