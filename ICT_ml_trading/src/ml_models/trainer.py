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
    prefix: str = 'grid',
    embargo: int = 0  # number of samples to embargo before each test fold
) -> Tuple[BaseEstimator, float, List[float]]:
    """
    📊 Nested TimeSeriesSplit CV with optional embargo:
    1️⃣ Outer loop for honest test performance (outer_splits)
    2️⃣ Inner loop for hyperparameter tuning (inner_splits)
    3️⃣ Embargo window to prevent forward‐lookahead leakage
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    outer_cv = TimeSeriesSplit(n_splits=outer_splits)
    inner_cv = TimeSeriesSplit(n_splits=inner_splits)
    scorer = get_scorer(scoring)

    outer_scores: List[float] = []
    best_params_list: List[Dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        # ── Apply embargo: drop the last `embargo` samples immediately before the test window
        if embargo > 0:
            first_test = test_idx.min()
            train_idx = train_idx[train_idx < first_test - embargo]

        # ── Log fold boundaries (datetime index if available, else integer)
        try:
            td = X.index[train_idx]
            vd = X.index[test_idx]
            print(f"TRAIN: {td.min()} → {td.max()},  TEST: {vd.min()} → {vd.max()}")
        except Exception:
            print(f"TRAIN idx: {train_idx.min()} → {train_idx.max()},  TEST idx: {test_idx.min()} → {test_idx.max()}")

        # ── Inner loop: hyperparameter search
        best_score, best_params = float('-inf'), {}
        for params in ParameterGrid(param_grid):
            cv_scores = []
            for inner_train, inner_val in inner_cv.split(train_idx):
                clf = clone(model).set_params(**params)
                clf.fit(
                    X.iloc[train_idx[inner_train]],
                    y.iloc[train_idx[inner_train]]
                )
                cv_scores.append(
                    scorer(
                        clf,
                        X.iloc[train_idx[inner_val]],
                        y.iloc[train_idx[inner_val]]
                    )
                )
            mean_cv = sum(cv_scores) / len(cv_scores)
            if mean_cv > best_score:
                best_score, best_params = mean_cv, params

        best_params_list.append(best_params)

        # ── Retrain on the full (embargoed) train split, evaluate on test split
        best_clf = clone(model).set_params(**best_params).fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )
        outer_score = scorer(best_clf, X.iloc[test_idx], y.iloc[test_idx])
        print(f"✅ Fold {fold} test {scoring}: {outer_score:.4f}")
        outer_scores.append(outer_score)

        # ── Checkpoint this fold’s best params
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"{prefix}_{model.__class__.__name__}_fold{fold}.pkl"
        )
        with open(ckpt_path, 'wb') as f:
            pickle.dump({'params': best_params}, f)

    avg_score = sum(outer_scores) / len(outer_scores)
    print(f"📊 Average nested CV {scoring}: {avg_score:.4f}")

    # ── Build final model on the full dataset using the most common params
    from collections import Counter
    most_common = Counter(
        tuple(sorted(p.items())) for p in best_params_list
    ).most_common(1)[0][0]
    final_params = dict(most_common)
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
