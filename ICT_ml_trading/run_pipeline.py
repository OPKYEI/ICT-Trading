# run_pipeline.py - Full pipeline with step icons, metrics, and fixed backtesting
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import joblib
from tqdm import tqdm
from ml_models.trainer import grid_search_with_checkpoint
# Visualization utilities
from utils.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_metric_bar
)
# ────────────────────────────────────────────────────────
# Make your `src/` folder importable
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ────────────────────────────────────────────────────────

# Import project modules after setting path
from data_processing.data_loader import DataLoader
from features.feature_engineering import ICTFeatureEngineer
from ml_models.model_builder import (
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    build_svm,
    build_xgboost
)
from trading.strategy import TradingStrategy
from trading.backtester import backtest_signals

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pipeline")


def load_checkpoint(checkpoint_dir, model_name, prefix):
    """Load model from checkpoint if it exists"""
    checkpoint_path = Path(checkpoint_dir) / f"{prefix}_{model_name}.joblib"
    if checkpoint_path.exists():
        try:
            model = joblib.load(checkpoint_path)
            print(f"✅ Loaded checkpoint for {model_name} from {checkpoint_path}")
            return model
        except Exception as e:
            print(f"❌ Failed to load checkpoint for {model_name}: {e}")
    return None


def evaluate_and_save_metrics(name, model, X_test, y_test, out_dir):
    """Evaluate model, print metrics with icons, return a dict, and append to CSV later"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    avg_prec = average_precision_score(y_test, y_prob) if y_prob is not None else None
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    print(f"\n======== Metrics for {name} ========")
    print(f"📊 Accuracy: {acc:.4f}" )
    print(f"📊 Precision (PPV): {ppv:.4f}" )
    print(f"📊 Recall (Sensitivity): {rec:.4f}" )
    print(f"📊 Specificity: {specificity:.4f}" )
    print(f"📊 F1 Score: {f1:.4f}" )
    if roc_auc is not None:
        print(f"📊 ROC AUC: {roc_auc:.4f}" )
    if avg_prec is not None:
        print(f"📊 Average Precision (NPV): {avg_prec:.4f}" )
    print(f"🧮 Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    metrics = {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1_score": f1,
        "npv": npv,
        "ppv": ppv,
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
    return metrics


def main():
    # ────────────────────────────────────────────────────────
    # STEP 1: Load Data
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 1: Loading data")
    csv_file = PROJECT_ROOT / "data" / "USDJPY=X_60m.csv"
    csv_name = csv_file.stem

    df = pd.read_csv(csv_file)
    if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    print(f"Loaded data with {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print("✅ STEP 1 complete: Data loaded")

    # ────────────────────────────────────────────────────────
    # STEP 2: Split Raw Data First
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 2: Splitting raw data (80/20)")
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # ────────────────────────────────────────────────────────
    # STEP 3: Feature Engineering on Separate Datasets
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 3: Feature engineering on separate datasets")
    symbol = csv_name.split('=')[0]
    fe = ICTFeatureEngineer(lookback_periods=[5, 10, 20], feature_selection_threshold=0.01)

    train_features = fe.engineer_features(train_data, symbol=symbol, additional_data={}).features
    test_features  = fe.engineer_features(test_data,  symbol=symbol, additional_data={}).features
    test_index     = test_data.index.copy()

    future_cols = [col for col in train_features.columns if 'future_' in col and col != 'future_direction_5']
    X_train = train_features.drop(columns=future_cols)
    y_train = X_train.pop('future_direction_5')
    X_test  = test_features.drop(columns=future_cols)
    y_test  = X_test.pop('future_direction_5')
    print("✅ STEP 3 complete: Features engineered")

    # ────────────────────────────────────────────────────────
    # STEP 4: Nested CV Training & Metrics 
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 4: Nested TimeSeriesSplit CV training")
    model_configs = {
        "rf": (build_random_forest(),      {"clf__n_estimators": [50, 100]}),
        "lr": (build_logistic_regression(), {"clf__C": [0.1, 1.0]}),
        "xgb":(build_xgboost(),            {"clf__n_estimators": [50, 100]}),
        "gb": (build_gradient_boosting(),  {"clf__n_estimators": [50]})
    }
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    trained_models = {}
    metrics_list    = []
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    for name, (model, params) in model_configs.items():
        print(f"\n📈 Nested CV for {name}")
        # Perform nested CV: 5 outer folds, 3 inner folds
        final_model, avg_score, fold_scores = grid_search_with_checkpoint(
            model=model,
            param_grid=params,
            X=X_train,
            y=y_train,
            outer_splits=5,
            inner_splits=3,
            scoring='accuracy',
            checkpoint_dir=str(checkpoint_dir),
            prefix=f"{csv_name}_{name}"
        )
        print(f"📊 {name} avg nested accuracy: {avg_score:.4f} (folds: {fold_scores})")
        trained_models[name] = final_model

        # Evaluate on holdout test
        metrics = evaluate_and_save_metrics(name, final_model, X_test, y_test, reports_dir)
        metrics_list.append(metrics)
        print(f"✅ {name} complete: Test accuracy {metrics['accuracy']:.4f}")

    # Save all model metrics
    pd.DataFrame(metrics_list).to_csv(reports_dir / f"{csv_name}_model_performance.csv", index=False)
    print("\n✅ STEP 4 complete: Nested CV training & metrics saved")


    # ────────────────────────────────────────────────────────
    # STEP 5: Evaluate Best Model & Shuffle Test
    # ────────────────────────────────────────────────────────
    best_name = max(metrics_list, key=lambda m: m['f1_score'])['model']
    best_model = trained_models[best_name]
    print(f"\n✅ STEP 5: Best model selected -> {best_name}")

    # 🔀 Label-Shuffle Stress Test
    print("\n🔀 Performing label-shuffle test")
    y_shuffled = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Clone the pipeline (retaining hyperparameters), then fit on shuffled labels
    shuffled_model = clone(best_model)
    shuffled_model.fit(X_train, y_shuffled)


    # ────────────────────────────────────────────────────────
    # STEP 6: Backtesting
    # ────────────────────────────────────────────────────────
    print("\n======== Backtesting Model ========")
    signals  = TradingStrategy(best_model).generate_signals(X_test)
    price_df = pd.DataFrame({'close': test_data['close']}, index=test_index)
    common_idx = signals.index.intersection(price_df.index)
    print(f"Common indices for backtest: {len(common_idx)} / {len(signals)}")

    if len(common_idx) > 0:
        aligned_signals = signals.loc[common_idx]
        aligned_prices  = price_df.loc[common_idx]
        equity = backtest_signals(aligned_signals, aligned_prices, initial_equity=10_000).to_frame()

        # Compute backtest metrics
        final_equity = equity.iloc[-1]['equity']
        returns      = (final_equity - 10_000) / 10_000 * 100
        max_dd       = ((equity['equity'].cummax() - equity['equity']) / equity['equity'].cummax()).max() * 100

        print(f"💰 Final Equity: ${final_equity:.2f} ({returns:.2f}% return)")
        print(f"📉 Max Drawdown: {max_dd:.2f}%")

        # Save equity curve
        eq_file = reports_dir / f"{csv_name}_equity_curve_{best_name}.csv"
        equity.to_csv(eq_file)
        print(f"✅ STEP 6 complete: Equity curve saved to {eq_file}")
    else:
        print("⚠️ STEP 6 skipped: No overlapping indices for backtesting")

    # ────────────────────────────────────────────────────────
    # STEP 7: Completion
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 7: Pipeline completed successfully.")


if __name__ == "__main__":
    main()
