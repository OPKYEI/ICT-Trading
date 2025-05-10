# run_pipeline.py — Full pipeline with step icons, metrics, and fixed regime walk‐forward

import warnings
warnings.filterwarnings("ignore")

# scikit-learn pipeline & tools
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone as clone_estimator, clone

# model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# data handling & utilities
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

# your project modules
from src.utils.config import BROKER_NAME, PIP_SIZE_DICT, DEFAULT_PIP_SIZE, USE_TP_SL
from ml_models.trainer import grid_search_with_checkpoint
from utils.visualization import plot_equity_curve, plot_drawdown, plot_metric_bar
from run_pipeline_extensions import regime_walk_forward, monte_carlo_bootstrap
from src.data_processing.data_loader import DataLoader
from features.feature_engineering import ICTFeatureEngineer, ICTFeatureTransformer
from ml_models.model_builder import (
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    build_svm,
    build_xgboost
)
from trading.strategy import TradingStrategy
from trading.backtester import backtest_signals


# ────────────────────────────────────────────────────────
# Make `src/` folder importable
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ────────────────────────────────────────────────────────

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

    metrics_df = pd.DataFrame([{    
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "specificity": spec, "f1_score": f1,
        "npv": npv, "ppv": ppv, "roc_auc": roc,
        "average_precision": ap, "tp": tp, "tn": tn,
        "fp": fp, "fn": fn
    }])
    metrics_df.to_csv(Path(out_dir)/f"{name}_metrics.csv", index=False)

    return metrics_df.to_dict(orient="records")[0]


def main():
    # STEP 1: Load Data
    print("\n✅ STEP 1: Loading data")
    csv_file = PROJECT_ROOT / "data" / "XAGUSD=X_60m.csv"
    csv_name = csv_file.stem
    symbol = csv_name.split('=')[0] if '=' in csv_name else csv_name.split('_')[0]

    data_loader = DataLoader(data_path=PROJECT_ROOT / "data")
    try:
        df = data_loader._load_local_data(
            symbol=symbol,
            start_date=None,
            end_date=None,
            interval="60m"
        )
        if df is None or df.empty:
            print("⚠️ DataLoader fallback to pd.read_csv")
            df = pd.read_csv(csv_file)
            if "timestamp" in df.columns:
                df.rename(columns={"timestamp": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
            df.set_index("datetime", inplace=True)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

    required_cols = ['open', 'high', 'low', 'close']
    missing = [c for c in required_cols if c not in map(str.lower, df.columns)]
    if missing:
        raise ValueError(f"Missing cols: {missing}")
    df.columns = [c.lower() for c in df.columns]

    print(f"Loaded data {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    print("✅ STEP 1 complete: Data loaded")

    # STEP 2: Create train/validation + never-touch hold-out
    print("\n✅ STEP 2: Creating hold-out and validation splits")

    # 2.1 Hold out the last 10%
    holdout_size = int(len(df) * 0.10)
    holdout_data = df.iloc[-holdout_size:]
    df_main      = df.iloc[:-holdout_size]
    print(f"Held out {len(holdout_data)} rows: "
          f"{holdout_data.index[0]} → {holdout_data.index[-1]}")

    # 2.2 On the remaining 90%, do an 80/20 train/validation split
    split_i    = int(len(df_main) * 0.8)
    train_data = df_main.iloc[:split_i]
    test_data  = df_main.iloc[split_i:]
    print(f"Train: {len(train_data)}, Validation: {len(test_data)}")

    pip_size = PIP_SIZE_DICT.get(symbol, DEFAULT_PIP_SIZE)

    
    # ────────────────────────────────────────────────────────
    # STEP 3: Compute features/targets & build Pipeline (with imputer)
    print("\n✅ STEP 3: Computing features/targets & building Pipeline")

    # 3.1 Full‐run feature engineering to extract targets
    fe_temp = ICTFeatureEngineer(
        lookback_periods=[5, 10, 20],
        feature_selection_threshold=0.01
    )
    fs_train = fe_temp.engineer_features(data=train_data, symbol=symbol, additional_data={})
    train_feats = fs_train.features
    fs_test  = fe_temp.engineer_features(data=test_data,  symbol=symbol, additional_data={})
    test_feats  = fs_test.features

    # 3.2 Extract targets and align timestamps
    y_train = train_feats["future_direction_5"].dropna()
    y_test  = test_feats ["future_direction_5"].dropna()
    X_raw_train = train_data.loc[y_train.index]
    X_raw_test  = test_data .loc[y_test.index]

    print(f"Aligned train samples: {len(X_raw_train)}, test samples: {len(X_raw_test)}")

    # 3.3 Build Pipeline: feature transformer → imputer → classifier
    pipeline = Pipeline([
        ("features", ICTFeatureTransformer(
            lookback_periods=[5, 10, 20],
            feature_selection_threshold=0.01,
            symbol=symbol,
            additional_data={}
        )),
        ("imputer", SimpleImputer(strategy="median")),     # ← IMPUTE FIRST
        ("selector", SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            threshold="median"
        )),
        ("clf", build_random_forest())                       # ← THEN SELECT & CLASSIFY
    ])


    print("✅ STEP 3 complete: Pipeline (with imputer) ready")

    #print("Pipeline steps:", pipeline.named_steps)
    #print("Pipeline parameter names:", list(pipeline.get_params().keys()))

    model_configs = {
        #"rf":  (build_random_forest,       {"clf__clf__n_estimators": [50, 100]}),
        #"lr":  (build_logistic_regression, {"clf__clf__C": [0.1, 1.0]}),
        "xgb": (build_xgboost,             {"clf__clf__n_estimators": [50, 100]}),
        #"gb":  (build_gradient_boosting,   {"clf__clf__n_estimators": [50]})
    }
    checkpoint_dir = PROJECT_ROOT / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    reports_dir    = PROJECT_ROOT / "reports";    reports_dir.mkdir(exist_ok=True)

    
    #_________________________________________________________
    # 
    # STEP 4: Nested TimeSeriesSplit CV training via Pipeline
    #_________________________________________________________
    
    print("\n✅ STEP 4: Nested TimeSeriesSplit CV training")

    trained_models = {}
    metrics_list = []

    for name, (builder_fn, params) in model_configs.items():
        print(f"\n📈 Nested CV for {name}")

        # inject the correct classifier into our pipeline
        pipeline.set_params(clf=builder_fn())

        final_model, avg_score, fold_scores = grid_search_with_checkpoint(
            model=pipeline,
            param_grid=params,            # e.g. {"clf__n_estimators": [50,100]}
            X=X_raw_train,
            y=y_train,
            outer_splits=5,
            inner_splits=3,
            scoring="accuracy",
            checkpoint_dir=str(checkpoint_dir),
            prefix=f"{csv_name}_{name}",
            embargo=25
        )

        print(f"📊 {name} avg nested accuracy: {avg_score:.4f} (folds: {fold_scores})")
        trained_models[name] = final_model

        # Evaluate on raw test set (features auto-created inside pipeline)
        m = evaluate_and_save_metrics(name, final_model, X_raw_test, y_test, reports_dir)
        metrics_list.append(m)
        print(f"✅ {name} complete: Test accuracy {m['accuracy']:.4f}")

        # Save pipeline for live‐trade
        pipeline_path = checkpoint_dir / f"{csv_name}_{name}_pipeline.pkl"
        joblib.dump(final_model, pipeline_path)
        print(f"✅ Saved trained pipeline '{name}' → {pipeline_path}")

    pd.DataFrame(metrics_list).to_csv(
        reports_dir / f"{csv_name}_model_performance.csv", index=False
    )
    print("\n✅ STEP 4 complete: Nested CV training & metrics saved")


    # STEP 5: Best Model & Shuffle Test
    best_name  = max(metrics_list, key=lambda x: x["f1_score"])["model"]
    best_model = trained_models[best_name]
    print(f"\n✅ STEP 5: Best model selected -> {best_name}")

    # Persist the single best model for live trading
    best_pipeline_path = checkpoint_dir / f"{csv_name}_best_pipeline_{best_name}.pkl"
    joblib.dump(best_model, best_pipeline_path)
    print(f"🚀 Saved BEST pipeline '{best_name}' → {best_pipeline_path}")
    #____________________________________________________________________
    #Step 6: Label Shuffle Test
    #____________________________________________________________________
    print("\n Step 6🔀: Performing label-shuffle test")
    # Shuffle labels only (preserve shape, drop index alignment)
    y_shuffled = y_train.sample(frac=1.0, random_state=42).values

    shuffled = clone(best_model)
    shuffled.fit(X_raw_train, y_shuffled)

    shuffled_metrics = evaluate_and_save_metrics(
        "shuffled", 
        shuffled, 
        X_raw_test, 
        y_test, 
        reports_dir
    )
    if shuffled_metrics["accuracy"] > 0.52:
        print("⚠️ Shuffled accuracy still high → leakage?")
    else:
        print("✅ Shuffled-label accuracy collapsed → minimal leakage")
        
    print("n\ Step 6: Label-shuffle test complete")
    
    # ────────────────────────────────────────────────────────
    # STEP 7: Evaluate on never-touch hold-out
    print("\n⏳ STEP 7: Evaluating on hold-out period")

    # 7.1 Compute features+targets for hold-out
    fe_hold = ICTFeatureEngineer(
        lookback_periods=[5,10,20],
        feature_selection_threshold=0.01
    )
    fs_hold   = fe_hold.engineer_features(data=holdout_data, symbol=symbol, additional_data={})
    feats_hold = fs_hold.features
    y_hold     = feats_hold["future_direction_5"].dropna()
    X_raw_hold = holdout_data.loc[y_hold.index]

    # 7.2 One-time hold-out evaluation
    metrics_hold = evaluate_and_save_metrics(
        "holdout",
        best_model,
        X_raw_hold,
        y_hold,
        reports_dir
    )
    print(f"✅ Hold-out accuracy: {metrics_hold['accuracy']:.4f}")
    print("✅ STEP 7 complete: Hold-out accuracy")
    
    # ────────────────────────────────────────────────────────
    # STEP 7.5: Shuffle-distribution test (30 runs)
    print("\n🎲 STEP 7.5: Running shuffle-distribution (30 shuffles)")
    shuffle_accs = []
    for seed in range(30):
        # shuffle labels
        y_shuf = y_train.sample(frac=1.0, random_state=seed).values
        mdl    = clone(best_model)
        mdl.fit(X_raw_train, y_shuf)
        shuffle_accs.append(mdl.score(X_raw_test, y_test))

    mean_sh, std_sh = np.mean(shuffle_accs), np.std(shuffle_accs)
    threshold = mean_sh + 3 * std_sh
    print(f"🔀 Shuffle mean: {mean_sh:.4f}, std: {std_sh:.4f}, mean+3σ: {threshold:.4f}")

    # STEP 8: Backtesting
    print("\n======== Step 8: Backtesting Model ========")
    # ────────────────────────────────────────────────────────
    # STEP 9: Backtesting Model
    signals = TradingStrategy(best_model).generate_signals(X_raw_test)
    print(f"🔍 Generated signals: {len(signals)} rows; unique: {signals.index.nunique()}")

    price_df = test_data[["high","low","close"]].loc[X_raw_test.index]
    common_idx = signals.index.intersection(price_df.index)


    if not common_idx.empty:
        aligned_signals = signals.loc[common_idx].loc[~signals.index.duplicated()]
        aligned_prices  = price_df.loc[common_idx].loc[~price_df.index.duplicated()]

        equity = backtest_signals(
            aligned_signals, aligned_prices,
            initial_equity=10_000.0, pip_size=pip_size
        ).to_frame("equity")

        final_eq = equity["equity"].iloc[-1]
        ret      = (final_eq - 10_000)/10_000 *100
        max_dd   = ((equity["equity"].cummax()-equity["equity"])/equity["equity"].cummax()).max()*100
        print(f"💰 Final Equity: ${final_eq:.2f} ({ret:.2f}% return)")
        print(f"📉 Max Drawdown: {max_dd:.2f}%")

        hr = equity["equity"].pct_change().dropna()
        sharpe = (hr.mean()/hr.std()) * np.sqrt(252*24)
        print(f"📈 Annualized Sharpe (hourly): {sharpe:.2f}")

        # Extract trades PnL
        pos = aligned_signals["signal"].shift(1).fillna(0)
        changes = pos.diff().abs()>0
        events = equity.index[changes]
        trade_pnls = [ equity.at[e1,"equity"] - equity.at[e0,"equity"]
                       for e0,e1 in zip(events[::2], events[1::2]) ]
        trades = pd.Series(trade_pnls)
        n_trades = len(trades)
        win_rate = (trades>0).mean() if n_trades else np.nan
        avg_profit = trades[trades>0].mean() if win_rate>0 else 0.0
        avg_loss   = trades[trades<0].mean() if win_rate<1 else 0.0
        profit_factor = trades[trades>0].sum()/abs(trades[trades<0].sum()) if any(trades<0) else np.nan
        expectancy = trades.mean() if n_trades else np.nan
        avg_lot = changes.sum()/n_trades if n_trades else np.nan

        print(f"🔢 Trades: {n_trades}, 🎯 Win rate: {win_rate:.2%}")
        print(f"💵 Avg profit: ${avg_profit:.2f}, Avg loss: ${avg_loss:.2f}")
        print(f"⚖️ Profit factor: {profit_factor:.2f}, ✨ Expectancy: {expectancy:.2f}")
        print(f"📦 Avg lot size: {avg_lot:.2f}")

        plot_equity_curve(equity)
        plot_drawdown(equity)
        print("✅ STEP 8 complete: Equity and trade metrics")

        '''if BROKER_NAME.upper()=="OANDA":
            from trading.executor import OandaExecutor
            live_exec = OandaExecutor()
            prev_sig = 0

            # build a single DataFrame with both signal and price
            df_orders = pd.concat([
                aligned_signals.rename("signal"),
                aligned_prices[["close"]]
            ], axis=1)

            # now iterate over the two correct columns
            for ts, row in df_orders.iterrows():
                sig = int(row["signal"])
                price = float(row["close"])
                if sig != prev_sig:
                    live_exec.send_order(
                        timestamp=ts,
                        signal=sig,
                        price=price,
                        pip_size=pip_size
                    )
                prev_sig = sig'''

    else:
        print("⚠️ STEP 8 skipped: No overlapping indices")
    
    
    # STEP 9: Regime Walk-Forward & Monte Carlo
    builder_map = {
        "rf":  build_random_forest,
        "lr":  build_logistic_regression,
        "xgb": build_xgboost,
        "gb":  build_gradient_boosting
    }
    print("\n🧭 STEP 8: Regime Walk-Forward")
    regime_walk_forward(
        df_features=pd.concat([train_feats, test_feats]),
        df_price=df[["high","low","close"]],
        target=pd.concat([y_train, y_test]),
        model_builder=builder_map[best_name],
        n_windows=5,
        initial_equity=10_000.0,
        pip_size=pip_size
    )

    print("\n🎲 STEP 9: Monte Carlo Bootstrap")
    monte_carlo_bootstrap(equity["equity"], n_sims=1000, initial_equity=10_000.0)

    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    main()
