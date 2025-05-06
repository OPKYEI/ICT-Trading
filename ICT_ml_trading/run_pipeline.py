# run_pipeline.py – clean version using Logistic Regression, Random Forest, Gradient Boosting, SVM

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from data_processing.data_loader import DataLoader
from features.feature_engineering import ICTFeatureEngineer
from ml_models.model_builder import (
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    build_svm
)
from ml_models.trainer import train_multiple_models_with_split
from trading.strategy import TradingStrategy
from trading.backtester import backtest_signals

# ────────────────────────────────────────────────────────
# Make your `src/` folder importable
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ────────────────────────────────────────────────────────

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pipeline")

print("✅ STEP 1: Loading data")
loader = DataLoader(data_path=PROJECT_ROOT / "data")

csv_file = PROJECT_ROOT / "data" / "XAUUSD=X_15m.csv"
csv_name = csv_file.stem
df = pd.read_csv(csv_file)
df.rename(columns={"timestamp": "datetime"}, inplace=True)
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)

print("✅ STEP 1 complete: Data loaded")

print("✅ STEP 2: Starting feature engineering")
fe = ICTFeatureEngineer(lookback_periods=[5, 10, 20], feature_selection_threshold=0.01)
fs = fe.engineer_features(df, symbol="XAUUSD", additional_data={})
X = fs.features

if "future_direction_5" not in X.columns:
    raise ValueError("Missing target column: future_direction_5")
y = X.pop("future_direction_5")

print("✅ STEP 2 complete: Features engineered")

print("✅ STEP 3: Splitting train/test data (80/20)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

print("✅ STEP 4: Training multiple models")
model_configs = {
    "rf": (build_random_forest(), {"clf__n_estimators": [50, 100]}),
    "lr": (build_logistic_regression(), {"clf__C": [0.1, 1.0]}),
    "svm": (build_svm(), {"clf__C": [0.5, 1.0]}),
    "gb": (build_gradient_boosting(), {"clf__n_estimators": [50]})
}


trained_models = train_multiple_models_with_split(
    model_configs,
    X_train,
    y_train,
    checkpoint_dir=str(PROJECT_ROOT / "checkpoints"),
    checkpoint_prefix=csv_name
)

print("✅ STEP 5: Evaluating and backtesting best model")
best_name, best_info = max(trained_models.items(), key=lambda item: item[1]["test_accuracy"])
best_model = best_info["model"]

# Predict & report
print(f"Best model: {best_name}")
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Backtesting
price_df = df.loc[df.index.intersection(X_test.index)][["close"]]
signals = TradingStrategy(best_model).generate_signals(X_test)
equity = backtest_signals(signals, price_df, initial_equity=10_000)

print("✅ STEP 6: Saving equity curve")
output_dir = PROJECT_ROOT / "reports"
output_dir.mkdir(exist_ok=True)
equity.to_csv(output_dir / f"{csv_name}_equity_curve_{best_name}.csv")
print("Equity curve saved to:", output_dir / f"equity_curve_{best_name}.csv")
