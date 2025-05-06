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

print("✅ STEP 2: Splitting raw data (80/20)")
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]
print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

print("✅ STEP 3: Feature engineering on separate datasets")
fe = ICTFeatureEngineer(lookback_periods=[5, 10, 20], feature_selection_threshold=0.01)
train_features = fe.engineer_features(train_data, symbol="XAUUSD", additional_data={}).features
test_features = fe.engineer_features(test_data, symbol="XAUUSD", additional_data={}).features

# Remove future-looking features except our target
future_cols = [col for col in train_features.columns if 'future_' in col and col != 'future_direction_5']
X_train = train_features.drop(columns=future_cols)
y_train = X_train.pop('future_direction_5')

X_test = test_features.drop(columns=future_cols)
y_test = X_test.pop('future_direction_5')

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
price_df = test_data[["close"]]
signals = TradingStrategy(best_model).generate_signals(X_test)
# Ensure signals and price_df have exactly the same index
signals = signals.loc[signals.index.intersection(price_df.index)]
price_df = price_df.loc[price_df.index.intersection(signals.index)]
equity = backtest_signals(signals, price_df, initial_equity=10_000)

print("✅ STEP 6: Saving equity curve")
output_dir = PROJECT_ROOT / "reports"
output_dir.mkdir(exist_ok=True)
equity.to_csv(output_dir / f"{csv_name}_equity_curve_{best_name}.csv")
print("Equity curve saved to:", output_dir / f"equity_curve_{best_name}.csv")
