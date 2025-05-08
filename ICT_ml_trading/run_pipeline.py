# run_pipeline.py — Full pipeline with step icons, metrics, and fixed regime walk‐forward

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import joblib
from tqdm import tqdm
from src.utils.config import (
    BROKER_NAME, PIP_SIZE_DICT, DEFAULT_PIP_SIZE, USE_TP_SL
)
# custom utilities
from ml_models.trainer import grid_search_with_checkpoint
from utils.visualization import plot_equity_curve, plot_drawdown, plot_metric_bar
from run_pipeline_extensions import regime_walk_forward, monte_carlo_bootstrap
from pandas import Timestamp
# ────────────────────────────────────────────────────────
# Make `src/` folder importable
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ────────────────────────────────────────────────────────

# Project modules
from features.feature_engineering import ICTFeatureEngineer
from ml_models.model_builder import (
    build_logistic_regression, build_random_forest,
    build_gradient_boosting, build_svm, build_xgboost
)
from trading.strategy import TradingStrategy
from trading.backtester import backtest_signals
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pipeline")


def load_checkpoint(checkpoint_dir, model_name, prefix):
    """Load model from checkpoint if it exists"""
    path = Path(checkpoint_dir) / f"{prefix}_{model_name}.joblib"
    if path.exists():
        try:
            mdl = joblib.load(path)
            print(f"✅ Loaded checkpoint for {model_name} from {path}")
            return mdl
        except Exception as e:
            print(f"❌ Failed to load checkpoint for {model_name}: {e}")
    return None


def evaluate_and_save_metrics(name, model, X_test, y_test, out_dir):
    """Evaluate model, print metrics with icons, return metrics dict"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    ap   = average_precision_score(y_test, y_prob) if y_prob is not None else None
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    print(f"\n======== Metrics for {name} ========")
    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📊 Precision (PPV): {ppv:.4f}")
    print(f"📊 Recall  (Sensitivity): {rec:.4f}")
    print(f"📊 Specificity: {spec:.4f}")
    print(f"📊 F1 Score: {f1:.4f}")
    if roc is not None:
        print(f"📊 ROC AUC: {roc:.4f}")
    if ap is not None:
        print(f"📊 Avg Precision (NPV): {ap:.4f}")
    print(f"🧮 Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    pd.DataFrame([{    
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "specificity": spec, "f1_score": f1,
        "npv": npv, "ppv": ppv, "roc_auc": roc,
        "average_precision": ap, "tp": tp, "tn": tn,
        "fp": fp, "fn": fn
    }]).to_csv(Path(out_dir)/f"{name}_metrics.csv", index=False)

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "specificity": spec, "f1_score": f1,
            "npv": npv, "ppv": ppv, "roc_auc": roc,
            "average_precision": ap, "tp": tp, "tn": tn,
            "fp": fp, "fn": fn}


def main():
    # ────────────────────────────────────────────────────────
    # STEP 1: Load Data
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 1: Loading data")
    csv_file = PROJECT_ROOT / "data" / "AUDUSD=X_60m.csv"
    csv_name = csv_file.stem

    df = pd.read_csv(csv_file)
    if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M:%S.%f", dayfirst=True)
    df.set_index("datetime", inplace=True)
    print(f"Loaded data with {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print("✅ STEP 1 complete: Data loaded")

    # ────────────────────────────────────────────────────────
    # STEP 2: Split Raw Data First
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 2: Splitting raw data (80/20)")
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data  = df.iloc[train_size:]
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # determine pip size from config
    symbol   = csv_name.split('=')[0]
    pip_size = PIP_SIZE_DICT.get(symbol, DEFAULT_PIP_SIZE)

    # ────────────────────────────────────────────────────────
    # STEP 3: Feature Engineering on Separate Datasets
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 3: Feature engineering on separate datasets")
    
    # Initialize the feature engineer
    fe = ICTFeatureEngineer(lookback_periods=[5, 10, 20],
                          feature_selection_threshold=0.01)
    
    # Now run feature engineering normally
    print("🔧 Engineering features...")
    train_feats = fe.engineer_features(train_data, symbol=symbol,
                                     additional_data={}).features
    test_feats = fe.engineer_features(test_data, symbol=symbol,
                                    additional_data={}).features
                                    
    # ────────────────────────────────────────────────────────
    # Diagnostics: raw vs. unique row counts
    # ────────────────────────────────────────────────────────
    print(f"🔍 Raw train_data rows: {len(train_data)};  "
          f"Feature-engineered rows: {len(train_feats)};  "
          f"Unique timestamps: {train_feats.index.nunique()}")
    print(f"🔍 Raw test_data rows:  {len(test_data)};  "
          f"Feature-engineered rows: {len(test_feats)};  "
          f"Unique timestamps: {test_feats.index.nunique()}")

    # Prepare X/y
    future_cols = [c for c in train_feats.columns
                   if c.startswith("future_") and c != "future_direction_5"]
    X_train = train_feats.drop(columns=future_cols)
    y_train = X_train.pop("future_direction_5")
    X_test  = test_feats.drop(columns=future_cols)
    y_test  = X_test.pop("future_direction_5")
    print("✅ STEP 3 complete: Features engineered")

    # ────────────────────────────────────────────────────────
    # STEP 4: Nested CV Training & Metrics
    # ────────────────────────────────────────────────────────
    print("\n✅ STEP 4: Nested TimeSeriesSplit CV training")
    model_configs = {
        "rf":  (build_random_forest(),      {"clf__n_estimators": [50, 100]}),
        "lr":  (build_logistic_regression(),{"clf__C": [0.1, 1.0]}),
        "xgb": (build_xgboost(),            {"clf__n_estimators": [50, 100]}),
        "gb":  (build_gradient_boosting(),  {"clf__n_estimators": [50]})
    }
    checkpoint_dir = PROJECT_ROOT / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    reports_dir    = PROJECT_ROOT / "reports";    reports_dir.mkdir(exist_ok=True)

    trained_models = {}
    metrics_list   = []

    for name, (model, params) in model_configs.items():
        print(f"\n📈 Nested CV for {name}")
        final_model, avg_score, fold_scores = grid_search_with_checkpoint(
            model=model,
            param_grid=params,
            X=X_train,
            y=y_train,
            outer_splits=5,
            inner_splits=3,
            scoring="accuracy",
            checkpoint_dir=str(checkpoint_dir),
            prefix=f"{csv_name}_{name}"
        )
        print(f"📊 {name} avg nested accuracy: {avg_score:.4f} (folds: {fold_scores})")
        trained_models[name] = final_model

        m = evaluate_and_save_metrics(name, final_model,
                                      X_test, y_test,
                                      reports_dir)
        metrics_list.append(m)
        print(f"✅ {name} complete: Test accuracy {m['accuracy']:.4f}")

    pd.DataFrame(metrics_list).to_csv(
        reports_dir / f"{csv_name}_model_performance.csv", index=False)
    print("\n✅ STEP 4 complete: Nested CV training & metrics saved")

    # ────────────────────────────────────────────────────────
    # STEP 5: Evaluate Best Model & Shuffle Test
    # ────────────────────────────────────────────────────────
    best_name  = max(metrics_list, key=lambda x: x["f1_score"])["model"]
    best_model = trained_models[best_name]
    print(f"\n✅ STEP 5: Best model selected -> {best_name}")

    print("\n🔀 Performing label-shuffle test")
    y_shuffled = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffled = clone(best_model)
    shuffled.fit(X_train, y_shuffled)

    shuffled_metrics = evaluate_and_save_metrics("shuffled", shuffled,
                                                 X_test, y_test,
                                                 reports_dir)
    metrics_list.append(shuffled_metrics)
    if shuffled_metrics["accuracy"] > 0.6:
        print("⚠️ Shuffled-label accuracy still high → potential leakage remains")
    else:
        print("✅ Shuffled-label accuracy collapsed → minimal residual leakage")

    # ────────────────────────────────────────────────────────
    # STEP 6: Backtesting Model + True Trade Metrics
    # ────────────────────────────────────────────────────────
    print("\n======== Backtesting Model ========")
    signals = TradingStrategy(best_model).generate_signals(X_test)
    # Diagnostics: signals rows
    print(f"🔍 Generated signals: {len(signals)} rows;  "
          f"Unique timestamps: {signals.index.nunique()};  "
          f"Duplicate signal rows: {len(signals) - signals.index.nunique()}")

    
    # align price to X_test index
    price_df = test_data[["high", "low", "close"]].loc[X_test.index]

    common_idx = signals.index.intersection(price_df.index)
    print(f"Common indices for backtest: {len(common_idx)} / {len(signals)}")

    if not common_idx.empty:
        # select only overlapping timestamps
        aligned_signals = signals.loc[common_idx]
        aligned_prices  = price_df.loc[common_idx]

        # DROP any
        aligned_signals = aligned_signals[~aligned_signals.index.duplicated(keep='first')]
        aligned_prices  = aligned_prices[~aligned_prices.index.duplicated(keep='first')]

        # get equity curve
        equity = backtest_signals(
            aligned_signals,
            aligned_prices,
            initial_equity=10_000.0,
            pip_size=pip_size
        ).to_frame(name="equity")

        # basic stats
        final_eq = equity["equity"].iloc[-1]
        ret      = (final_eq - 10_000.0) / 10_000.0 * 100
        max_dd   = ((equity["equity"].cummax() - equity["equity"]) /
                    equity["equity"].cummax()).max() * 100
        print(f"💰 Final Equity: ${final_eq:.2f} ({ret:.2f}% return)")
        print(f"📉 Max Drawdown: {max_dd:.2f}%")

        # Sharpe (hourly → annualized)
        hr = equity["equity"].pct_change().dropna()
        sharpe = (hr.mean() / hr.std()) * np.sqrt(252 * 24)
        print(f"📈 Annualized Sharpe (hourly): {sharpe:.2f}")

        # now extract *actual* trades from equity
        pos = aligned_signals["signal"].shift(1).fillna(0)
        changes = pos.diff().abs() > 0
        events = equity.index[changes]

        # pair entry→exit for round-trip PnL
        trade_pnls = []
        for i in range(0, len(events) - 1, 2):
            e0, e1 = events[i], events[i+1]
            pnl = equity.at[e1, "equity"] - equity.at[e0, "equity"]
            trade_pnls.append(pnl)

        trades = pd.Series(trade_pnls)
        n_trades = len(trades)
        win_rate = (trades > 0).mean() if n_trades else np.nan
        avg_profit = trades[trades > 0].mean() if win_rate > 0 else 0.0
        avg_loss   = trades[trades < 0].mean() if win_rate < 1 else 0.0
        profit_factor = trades[trades > 0].sum() / abs(trades[trades < 0].sum()) if any(trades < 0) else np.nan
        expectancy    = trades.mean() if n_trades else np.nan
        avg_lot_size  = changes.sum() / n_trades if n_trades else np.nan

        print(f"🔢 Number of trades: {n_trades}")
        print(f"🎯 Win rate: {win_rate:.2%}")
        print(f"💵 Avg profit ($): {avg_profit:.2f}, Avg loss ($): {avg_loss:.2f}")
        print(f"⚖️ Profit factor: {profit_factor:.2f}")
        print(f"✨ Expectancy per trade: {expectancy:.2f}")
        print(f"📦 Avg lot size per trade: {avg_lot_size:.2f}")

        # Visualize
        plot_equity_curve(equity)
        plot_drawdown(equity)
        print("✅ STEP 6 complete: Equity and trade metrics")

    else:
        print("⚠️ STEP 6 skipped: No overlapping indices for backtesting")

    # ────────────────────────────────────────────────────────
    # STEP 7: Regime Walk-Forward & Monte Carlo Bootstrap
    # ────────────────────────────────────────────────────────
    builder_map = {
        "rf":  build_random_forest,
        "lr":  build_logistic_regression,
        "xgb": build_xgboost,
        "gb":  build_gradient_boosting
    }

    print("\n🧭 STEP 7: Regime Walk-Forward")
    regime_walk_forward(
        df_features=pd.concat([train_feats, test_feats]),
        df_price=df[["high", "low", "close"]],
        target=pd.concat([y_train, y_test]),
        model_builder=builder_map[best_name],
        n_windows=5,
        initial_equity=10_000.0,
        pip_size=pip_size
    )

    print("\n🎲 STEP 8: Monte Carlo Bootstrap")
    monte_carlo_bootstrap(equity["equity"], n_sims=1000, initial_equity=10_000.0)

    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()