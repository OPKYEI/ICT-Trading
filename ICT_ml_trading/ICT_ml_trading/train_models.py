# run_pipeline.py — Full pipeline with multi-file processing

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
import glob

# your project modules
from src.ml_models.trainer import load_checkpoint
from sklearn.preprocessing import FunctionTransformer
from src.utils.config import BROKERS, PIP_SIZE_DICT, DEFAULT_PIP_SIZE, USE_TP_SL
from src.ml_models.trainer import grid_search_with_checkpoint
from src.utils.visualization import plot_equity_curve, plot_drawdown, plot_metric_bar
from run_pipeline_extensions import regime_walk_forward, monte_carlo_bootstrap
from src.data_processing.data_loader import DataLoader
from src.features.feature_engineering import ICTFeatureEngineer, ICTFeatureTransformer
from src.ml_models.model_builder import (
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    build_svm,
    build_xgboost
)
from src.trading.strategy import TradingStrategy
from src.trading.backtester import backtest_signals

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


def get_csv_files_to_process(data_dir, checkpoint_dir):
    """
    Find all CSV files in data directory and filter out those already processed.
    A file is considered "processed" if it has a BEST model saved.
    
    Returns:
        list: CSV files that need processing
    """
    # Find all CSV files
    csv_pattern = str(data_dir / "*.csv")
    all_csv_files = glob.glob(csv_pattern)
    csv_files = [Path(f) for f in all_csv_files]
    
    print(f"📁 Found {len(csv_files)} CSV files in {data_dir}")
    
    # Check which ones need processing
    files_to_process = []
    
    for csv_file in csv_files:
        csv_name = csv_file.stem
        
        # Look for BEST model pattern: {csv_name}_best_pipeline_*.pkl
        best_model_pattern = f"{csv_name}_best_pipeline_*.pkl"
        best_model_files = list(checkpoint_dir.glob(best_model_pattern))
        
        if not best_model_files:
            # No best model found - needs processing
            files_to_process.append(csv_file)
            print(f"📋 Queued for processing: {csv_name} (no best model found)")
        else:
            # Best model exists - skip
            best_model_file = best_model_files[0].name
            print(f"✅ Already processed (skipping): {csv_name} → {best_model_file}")
    
    print(f"\n🎯 Total files to process: {len(files_to_process)}")
    return files_to_process


def check_file_requirements(csv_file):
    """
    Check if CSV file meets basic requirements for processing.
    
    Returns:
        tuple: (is_valid, reason)
    """
    try:
        # Quick check - load first few rows
        df_sample = pd.read_csv(csv_file, nrows=100, parse_dates=[0], index_col=0)
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_sample.columns]
        
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check if enough data
        df_full_size = len(pd.read_csv(csv_file))
        if df_full_size < 1000:  # Minimum rows needed
            return False, f"Insufficient data: {df_full_size} rows (need >1000)"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


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


def to_numpy_array(X):
    """Convert pandas DataFrame to numpy array for the pipeline."""
    return X.values


def process_single_file(csv_file, checkpoint_dir, reports_dir, data_loader):
    """
    Process a single CSV file through the entire ML pipeline.
    NOW WITH ENHANCED BACKTEST METRICS!
    
    Returns:
        dict: Processing results and metrics
    """
    csv_name = csv_file.stem
    symbol = csv_name.split('=')[0] if '=' in csv_name else csv_name.split('_')[0]
    
    print(f"\n{'='*60}")
    print(f"🚀 PROCESSING: {csv_name}")
    print(f"📈 Symbol: {symbol}")
    print(f"{'='*60}")
    
    try:
        # STEP 1: Load & align multi‐timeframe data
        print(f"\n✅ STEP 1: Loading multi‐TF data for {csv_name}")
        
        # Returns (hourly_df, {"1D": daily_df, "1W": weekly_df})
        df, extras_full = data_loader.load_multi_timeframe(
            csv_path=csv_file,
            base_tf="1H",
            extra_tfs=["1D", "1W"]
        )

        print(f"Loaded & aligned data {len(df)} rows from {df.index[0]} to {df.index[-1]}")

        # STEP 2: Create train/validation + never-touch hold-out
        print(f"\n✅ STEP 2: Creating hold-out and validation splits for {csv_name}")
        holdout_size = int(len(df) * 0.10)
        holdout_data = df.iloc[-holdout_size:]
        df_main      = df.iloc[:-holdout_size]
        
        split_i    = int(len(df_main) * 0.8)
        train_data = df_main.iloc[:split_i]
        test_data  = df_main.iloc[split_i:]
        print(f"Train: {len(train_data)}, Validation: {len(test_data)}, Holdout: {len(holdout_data)}")

        # STEP 3: Compute features/targets & build Pipeline
        print(f"\n✅ STEP 3: Computing features/targets & building Pipeline for {csv_name}")

        # 3.1 Full‐run feature engineering to extract targets
        fe_temp = ICTFeatureEngineer(
            lookback_periods=[5, 10, 20],
            feature_selection_threshold=0.01
        )
        fs_train = fe_temp.engineer_features(
            data=train_data,
            symbol=symbol,
            additional_data=extras_full
        )
        train_feats = fs_train.features

        fs_test = fe_temp.engineer_features(
            data=test_data,
            symbol=symbol,
            additional_data=extras_full
        )
        test_feats = fs_test.features

        # 3.2 Extract targets and align timestamps
        y_train = train_feats["future_direction_5"].dropna()
        y_test  = test_feats["future_direction_5"].dropna()
        X_raw_train = train_data.loc[y_train.index]
        X_raw_test  = test_data.loc[y_test.index]

        print(f"Aligned train samples: {len(X_raw_train)}, test samples: {len(X_raw_test)}")

        # 3.3 Build Pipeline
        pipeline = Pipeline([
            ("features", ICTFeatureTransformer(
                lookback_periods=[5, 10, 20],
                feature_selection_threshold=0.01,
                symbol=symbol,
                additional_data=extras_full
            )),
            ("to_numpy", FunctionTransformer(func=to_numpy_array, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectFromModel(
                estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                threshold="median"
            )),
            ("clf", build_random_forest())
        ])

        # STEP 4: Train multiple models
        print(f"\n✅ STEP 4: Nested TimeSeriesSplit CV training for {csv_name}")
        trained_models = {}
        metrics_list  = []

        model_configs = {
            "rf":  (build_random_forest,       {"clf__clf__n_estimators": [50, 100]}),
            "lr":  (build_logistic_regression, {"clf__clf__C": [0.1, 1.0]}),
            "xgb": (build_xgboost,             {"clf__clf__n_estimators": [50, 100]}),
            "gb":  (build_gradient_boosting,   {"clf__clf__n_estimators": [50]})
        }

        for name, (builder_fn, params) in model_configs.items():
            print(f"\n📈 Training {name} for {csv_name}")

            # inject the correct classifier into our pipeline
            pipeline.set_params(clf=builder_fn())

            # 1. Attempt to load an existing checkpoint
            ckpt_model = load_checkpoint(checkpoint_dir, name, prefix=f"{csv_name}_{name}")
            if ckpt_model is not None:
                final_model = ckpt_model
                print(f"⚡ Skipping CV, using loaded checkpoint for {name}")
                avg_score, fold_scores = None, []
            else:
                # 2. No checkpoint found → run nested CV
                final_model, avg_score, fold_scores = grid_search_with_checkpoint(
                    model=pipeline,
                    param_grid=params,
                    X=X_raw_train,
                    y=y_train,
                    outer_splits=5,
                    inner_splits=3,
                    scoring="accuracy",
                    checkpoint_dir=str(checkpoint_dir),
                    prefix=f"{csv_name}_{name}",
                    embargo=25
                )
                # 3. Save the best model
                ckpt_path = Path(checkpoint_dir) / f"{csv_name}_{name}.joblib"
                joblib.dump(final_model, ckpt_path)
                print(f"✅ Saved CV checkpoint for {name} → {ckpt_path}")

            if avg_score is None:
                print(f"📊 {name}: loaded from checkpoint, CV skipped")
            else:
                print(f"📊 {name} avg nested accuracy: {avg_score:.4f}")

            trained_models[name] = final_model

            # Evaluate on test set
            file_reports_dir = reports_dir / csv_name
            file_reports_dir.mkdir(exist_ok=True)
            
            m = evaluate_and_save_metrics(name, final_model, X_raw_test, y_test, file_reports_dir)
            metrics_list.append(m)
            print(f"✅ {name} complete: Test accuracy {m['accuracy']:.4f}")

            # Save pipeline for live‐trade
            pipeline_path = checkpoint_dir / f"{csv_name}_{name}_pipeline.pkl"
            joblib.dump(final_model, pipeline_path)

        # Save all metrics for this file
        pd.DataFrame(metrics_list).to_csv(
            file_reports_dir / f"{csv_name}_model_performance.csv", index=False
        )

        # STEP 5: Best Model Selection
        best_name  = max(metrics_list, key=lambda x: x["f1_score"])["model"]
        best_model = trained_models[best_name]
        print(f"\n✅ STEP 5: Best model for {csv_name} -> {best_name}")

        # Save best model
        best_pipeline_path = checkpoint_dir / f"{csv_name}_best_pipeline_{best_name}.pkl"
        joblib.dump(best_model, best_pipeline_path)
        print(f"🚀 Saved BEST pipeline for {csv_name} → {best_pipeline_path}")

        # STEP 6: ENHANCED BACKTEST WITH FULL METRICS! 🚀
        print(f"\n✅ STEP 6: Enhanced backtest with full metrics for {csv_name}")
        signals = TradingStrategy(best_model).generate_signals(X_raw_test)
        price_df = test_data[["high","low","close"]].loc[X_raw_test.index]
        common_idx = signals.index.intersection(price_df.index)
        
        backtest_metrics = {}
        final_eq = 10_000.0  # Default fallback
        
        if not common_idx.empty:
            pip_size = PIP_SIZE_DICT.get(symbol, DEFAULT_PIP_SIZE)
            aligned_signals = signals.loc[common_idx].loc[~signals.index.duplicated()]
            aligned_prices  = price_df.loc[common_idx].loc[~price_df.index.duplicated()]

            # 🚀 USE THE ENHANCED BACKTEST FUNCTION! 🚀
            from src.trading.backtester import enhanced_backtest_with_metrics
            
            equity_curve, backtest_metrics = enhanced_backtest_with_metrics(
                signals=aligned_signals,
                price_data=aligned_prices,
                symbol=csv_name,  # Use full csv_name for better identification
                initial_equity=10_000.0,
                pip_size=pip_size
            )

            final_eq = equity_curve.iloc[-1]
            
            # Save equity curve for later analysis
            equity_df = equity_curve.to_frame("equity")
            equity_df.to_csv(file_reports_dir / f"{csv_name}_equity_curve.csv")
            
            # Save backtest metrics separately
            backtest_df = pd.DataFrame([backtest_metrics])
            backtest_df.to_csv(file_reports_dir / f"{csv_name}_backtest_metrics.csv", index=False)
            
        else:
            print(f"⚠️ {csv_name}: No overlapping indices for backtesting")
            backtest_metrics = {
                "total_return": np.nan,
                "annual_return": np.nan,
                "max_drawdown": np.nan,
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "calmar_ratio": np.nan
            }

        print(f"✅ {csv_name} processing complete!")
        
        return {
            "file": csv_name,
            "status": "success",
            "best_model": best_name,
            "metrics": metrics_list,
            "final_equity": final_eq,
            "backtest_metrics": backtest_metrics  # 🚀 NOW INCLUDING FULL METRICS!
        }
        
    except Exception as e:
        print(f"❌ Error processing {csv_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "file": csv_name,
            "status": "failed",
            "error": str(e),
            "final_equity": None,
            "backtest_metrics": {}
        }


def main():
    """Main pipeline function that processes all CSV files."""
    
    # Setup directories
    data_dir = PROJECT_ROOT / "data"
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    reports_dir = PROJECT_ROOT / "reports"
    
    checkpoint_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Data directory: {data_dir}")
    print(f"💾 Checkpoint directory: {checkpoint_dir}")
    print(f"📊 Reports directory: {reports_dir}")
    
    # Initialize data loader once
    data_loader = DataLoader(data_path=data_dir)
    
    # Get files that need processing
    files_to_process = get_csv_files_to_process(data_dir, checkpoint_dir)
    
    if not files_to_process:
        print("🎉 All files have been processed! Nothing to do.")
        return
    
    # Process each file
    results = []
    successful_files = 0
    failed_files = 0
    
    print(f"\n🚀 Starting batch processing of {len(files_to_process)} files...")
    
    for i, csv_file in enumerate(files_to_process, 1):
        print(f"\n📋 Processing file {i}/{len(files_to_process)}")
        
        # Check file requirements first
        is_valid, reason = check_file_requirements(csv_file)
        if not is_valid:
            print(f"⚠️ Skipping {csv_file.name}: {reason}")
            results.append({
                "file": csv_file.stem,
                "status": "skipped",
                "reason": reason
            })
            continue
        
        # Process the file
        start_time = datetime.now()
        result = process_single_file(csv_file, checkpoint_dir, reports_dir, data_loader)
        end_time = datetime.now()
        
        result["processing_time"] = str(end_time - start_time)
        results.append(result)
        
        if result["status"] == "success":
            successful_files += 1
            print(f"✅ {csv_file.name} completed in {result['processing_time']}")
        else:
            failed_files += 1
            print(f"❌ {csv_file.name} failed")
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful_files}")
    print(f"❌ Failed: {failed_files}")
    print(f"📁 Total processed: {len(files_to_process)}")
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = reports_dir / f"batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Summary saved to: {summary_path}")
    
    # Show successful files
    if successful_files > 0:
        print(f"\n🎉 Successfully processed files:")
        for result in results:
            if result["status"] == "success":
                print(f"  📈 {result['file']} - Best: {result['best_model']}")
    
    # Show failed files
    if failed_files > 0:
        print(f"\n⚠️ Failed files:")
        for result in results:
            if result["status"] == "failed":
                print(f"  ❌ {result['file']} - Error: {result.get('error', 'Unknown')}")
    
    print(f"\n🎯 All done! Check {reports_dir} for detailed results.")


if __name__ == "__main__":
    main()