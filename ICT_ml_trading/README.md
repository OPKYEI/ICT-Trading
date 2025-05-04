# ICT-ML-Trading

An end-to-end Forex trading research and execution framework implementing ICT (Inner Circle Trader) concepts with machine learning integration, backtesting, execution simulation, performance metrics, and utilities.

---

## 📝 Overview

ICT-ML-Trading provides:

* **Data infrastructure**: load, validate, and resample OHLCV data across multiple timeframes.
* **ICT-derived features**: market structure, PD Arrays, liquidity, time sessions, patterns, and intermarket analysis.
* **Machine Learning pipeline**: feature engineering, model building (Logistic Regression, Random Forest, Gradient Boosting, SVM), hyperparameter search with checkpointing, evaluation metrics.
* **Trading system**: signal generation, risk management, backtesting, execution simulation, and performance metrics (Sharpe, Sortino, drawdown, Calmar).
* **Utilities**: configuration management (JSON/YAML), visualization (equity curves, drawdowns, metrics), and structured logging.

---

## ⚙️ Prerequisites

* **Python 3.10** (ensure consistency across environments)
* **pip**
* **Git**

---

## 🚀 Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/your-org/ict_ml_trading.git
cd ict_ml_trading
```

### 2. Editable install

Installs the package in editable mode, linking your local `src/` to Python imports.

```bash
pip install -e .
```

### 3. (Optional) Jupyter kernel

Register a dedicated kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name ict-ml-trading --display-name "ICT-ML-Trading"
```

Select **ICT-ML-Trading** in Jupyter for notebook sessions.

---

## ✅ Running Tests

From project root:

```bash
pytest --maxfail=1 --disable-warnings -q
```

All tests should pass (currently 90+). Uses headless Matplotlib backend (Agg) for visualization tests.

---

## 📂 Project Structure

```text
ict_ml_trading/
├── data/
│   ├── raw/                    # Raw OHLCV CSVs
│   ├── processed/              # Feature-enhanced data
│   └── models/                 # Serialized ML models
├── src/
│   ├── data_processing/
│   │   ├── data_loader.py      # Load & validate OHLCV
│   │   └── timeframe_manager.py# Multi-timeframe alignment
│   ├── features/
│   │   ├── market_structure.py # HH/HL, LH/LL, MSS
│   │   ├── pd_arrays.py        # Order blocks, FVGs, breakers
│   │   ├── liquidity.py        # Liquidity levels & stop runs
│   │   ├── time_features.py    # Sessions, kill zones, encodings
│   │   ├── patterns.py         # OTE, Turtle Soup, breaker patterns
│   │   └── intermarket.py      # Correlations, SMT divergences
│   ├── ml_models/
│   │   ├── model_builder.py    # Builds LR, RF, GB, SVM
│   │   ├── trainer.py          # Grid/random search + checkpointing
│   │   └── evaluator.py        # Accuracy, F1, ROC-AUC & reports
│   ├── trading/
│   │   ├── strategy.py         # Signals from model probabilities
│   │   ├── risk_manager.py     # Position sizing & stop-loss
│   │   ├── backtester.py       # Equity curve simulation
│   │   └── executor.py         # Trade log & PnL simulation
│   └── utils/
│       ├── config.py           # JSON/YAML loader with fallback parser
│       ├── visualization.py    # Equity, drawdown, metrics plots
│       └── logging_config.py   # Structured logging setup
├── tests/                      # Pytest suites mirroring src/
├── notebooks/                  # Jupyter demos (exploration & visualization)
├── requirements.txt            # Pin dependencies
├── setup.py                    # Editable install config
└── README.md                   # This file
```

---

## 🔧 Configuration

Place a `config.json` or `config.yaml` at project root:

```yaml
# config.yaml
data:
  raw_path: data/raw
  processed_path: data/processed

models:
  random_forest:
    n_estimators: 100
    max_depth: 5

logging:
  level: INFO
  file: logs/app.log
```

Load it:

```python
from utils.config import load_config
cfg = load_config("config.yaml")
```

---

## 📊 Usage Examples

### 1. Data Loading

```python
from data_processing.data_loader import load_ohlcv
df = load_ohlcv("data/raw/EURUSD.csv")
```

### 2. Feature Engineering

```python
from features.feature_engineering import ICTFeatureEngineer
fe = ICTFeatureEngineer(lookback_periods=[5,10,20], feature_selection_threshold=0.01)
fs = fe.engineer_features(df, symbol="EURUSD", additional_data={"DXY": dxy_df})
X = fs.features
```

### 3. Model Training & Evaluation

```python
from ml_models.model_builder import build_random_forest
from ml_models.trainer import grid_search_with_checkpoint
from ml_models.evaluator import evaluate_classification_model

model = build_random_forest(n_estimators=100, pca_components=None)
best_model, results_df = grid_search_with_checkpoint(
    model, {"clf__n_estimators": [50,100]}, X_train, y_train,
    cv=3, scoring='accuracy', checkpoint_path='chkpts/rf.pkl'
)
metrics = evaluate_classification_model(best_model, X_test, y_test, output_path='reports/rf_eval')
```

### 4. Backtesting & Execution

```python
from trading.backtester import backtest_signals
from trading.executor import Executor

equity = backtest_signals(signals_df, price_df, initial_equity=10000)
exec = Executor()
trade_log = exec.execute(signals_df, price_df)
```

### 5. Performance & Visualization

```python
from trading.performance import compute_performance_metrics, print_performance_metrics
from utils.visualization import plot_equity_curve, plot_drawdown, plot_metric_bar

metrics = compute_performance_metrics(equity, period_per_year=252)
print_performance_metrics(metrics)
plot_equity_curve(equity, output_path='reports/equity.png', show=True)
plot_drawdown(equity, output_path='reports/dd.png')
plot_metric_bar(metrics, output_path='reports/metrics.png')
```

---

## 🔮 Roadmap

* **Interactive dashboard** (Streamlit/Dash)
* **Real broker API integration**
* **Additional ML architectures** (XGBoost, LSTM)
* **CI/CD pipeline** (GitHub Actions)

---

*Happy trading!*
