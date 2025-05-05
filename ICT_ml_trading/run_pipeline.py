# run_pipeline.py

import os
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────
# 1) Make your `src/` folder importable without packaging
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ────────────────────────────────────────────────────────

# 2) Imports (exactly as your tests use them)
from data_processing.data_loader    import DataLoader
from features.feature_engineering   import ICTFeatureEngineer
from ml_models.model_builder        import ModelBuilder
from ml_models.trainer              import grid_search_with_checkpoint
from trading.strategy               import TradingStrategy
from trading.backtester             import backtest_signals

# 3) Load your **local** CSVs from `data/` (not `data/raw/`)
loader = DataLoader(data_path=PROJECT_ROOT / "data")

df = loader.load_data(
    symbol="EURUSD",
    start_date="2023-01-01",
    end_date="2023-01-10",
    interval="1h",
    data_source="local"   # force local-only
)

# 4) Feature engineering
fe = ICTFeatureEngineer(lookback_periods=[5,10,20], feature_selection_threshold=0.01)
fs = fe.engineer_features(df, symbol="EURUSD", additional_data={})
X = fs.features
y = X.pop("future_direction_5")  # your target

# 5) Train a tiny grid (uses your existing checkpoint logic)
model, grid_results = grid_search_with_checkpoint(
    build_random_forest(n_estimators=50, pca_components=None),
    {"clf__n_estimators": [50, 100]},
    X, y,
    cv=3,
    scoring="accuracy",
    checkpoint_path=PROJECT_ROOT / "checkpoints" / "rf_checkpoint.pkl"
)
print("Best model params:", model.get_params())

# 6) Generate signals & backtest
signals = TradingStrategy(model).generate_signals(X)
equity  = backtest_signals(signals, df[["close"]], initial_equity=10_000)
print("Final equity:", equity.iloc[-1])

# 7) Save equity curve for inspection
(out := PROJECT_ROOT / "reports").mkdir(exist_ok=True)
equity.to_csv(out / "equity_curve.csv")
print("Equity curve →", out / "equity_curve.csv")
