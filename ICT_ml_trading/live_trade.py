#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import ctypes
import logging
import warnings
import time
import schedule
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# 1) Enable full UTF-8 (including emojis) on Windows console
# ----------------------------------------------------------------------------

def enable_unicode_console():
    if sys.platform == "win32":
        try:
            # Switch console code page to UTF-8
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            # Reconfigure Python's std streams to UTF-8
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Invoke as early as possible
enable_unicode_console()

# ----------------------------------------------------------------------------
# 2) Manual logging configuration (bypassing logging_config to guarantee UTF-8)
# ----------------------------------------------------------------------------

# Ensure logs directory exists
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a named logger
logger = logging.getLogger("live_trade")
logger.setLevel(logging.INFO)

# Define formatter
datefmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt=datefmt
)

# Console handler (UTF-8)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler (UTF-8)
file_handler = logging.FileHandler(log_dir / "live_trade.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ----------------------------------------------------------------------------
# 3) Suppress unwanted warnings
# ----------------------------------------------------------------------------
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 4) Import configuration and initialize core components
# ----------------------------------------------------------------------------
from src.utils.config import (
    BROKER_NAME,
    SYMBOL,
    PIP_SIZE_DICT,
    DEFAULT_PIP_SIZE,
    OANDA_API_TOKEN,
    OANDA_ACCOUNT_ID,
)
from src.data_processing.data_loader import DataLoader
from src.features.feature_engineering import ICTFeatureEngineer
from src.trading.strategy import TradingStrategy
from src.trading.executor import Executor, OandaExecutor

# Initialize data loader
data_loader = DataLoader(data_path=Path("data"))

# Determine live vs simulation
LIVE_MODE = BROKER_NAME.upper() == "OANDA" and OANDA_API_TOKEN != "YOUR_OANDA_TOKEN"
logger.info(f"Starting in {'LIVE' if LIVE_MODE else 'SIMULATION'} mode")

# Initialize feature engineer
fe = ICTFeatureEngineer(
    lookback_periods=[5, 10, 20],
    feature_selection_threshold=0.01,
)

# Optional Oanda fetcher
try:
    from src.data_processing.oanda_data import OandaDataFetcher
    oanda_fetcher = (
        OandaDataFetcher()
        if LIVE_MODE else None
    )

    if oanda_fetcher:
        logger.info("Initialized OandaDataFetcher")
except ImportError:
    logger.warning("OandaDataFetcher not available; using local data only")
    oanda_fetcher = None

# Track last trade details
last_traded_signal = 0
last_traded_time = None

# ----------------------------------------------------------------------------
# 5) Model loading: prefer 'best' pipelines
# ----------------------------------------------------------------------------
import joblib
def to_numpy_array(X):
    """Convert pandas DataFrame to numpy array."""
    return X.values

def load_model(symbol: str):
    """
    Locate the best pipeline file (with 'best' in its name) under checkpoints/ matching the symbol,
    load it via joblib, and return the fitted estimator.
    """
    clean_symbol = symbol.replace("_", "")
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        logger.warning(
            f"Checkpoint directory not found; defaulting to dummy RandomForestClassifier"
        )
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=42)

    # Gather all *_pipeline*.pkl files for this symbol
    pipeline_files = list(
        checkpoint_dir.glob(f"*{clean_symbol}*_pipeline*.pkl")
    )
    if not pipeline_files:
        logger.warning(
            f"No pipeline files found for '{symbol}'; defaulting to dummy RandomForestClassifier"
        )
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=42)

    # Prefer files containing 'best'
    best_candidates = [f for f in pipeline_files if 'best' in f.name.lower()]
    if best_candidates:
        chosen_file = sorted(best_candidates)[-1]
    else:
        chosen_file = sorted(pipeline_files)[-1]

    logger.info(f"📦 Loading pipeline via joblib: {chosen_file.name}")
    model = joblib.load(chosen_file)
    logger.info(f"🔑 Loaded estimator: {type(model).__name__}")
    return model

# Instantiate global model using joblib loader
declared_model = load_model(SYMBOL)


# ----------------------------------------------------------------------------
# 6) Trading cycle and scheduling
# ----------------------------------------------------------------------------

def run_once():
    global last_traded_signal, last_traded_time
    logger.info("▶️ Running trading cycle")
    now = datetime.now()

    # 1) Fetch OHLC data (live or local)
    df = None
    if LIVE_MODE and oanda_fetcher:
        try:
            sym = SYMBOL if "_" in SYMBOL else f"{SYMBOL[:3]}_{SYMBOL[3:]}"
            # Pull 5 days * 24 hours = 120 bars for feature lookbacks
            count = 5 * 24
            df = oanda_fetcher.fetch_ohlc(
                instrument=sym, granularity="H1", count=count
            )
            logger.info(f"Fetched {len(df)} bars (~{count}h) from Oanda")
        except Exception as e:
            logger.warning(f"Oanda fetch failed: {e}")

    if df is None or df.empty:
        loader_sym = SYMBOL.replace("_", "")
        df = data_loader.load_data(
            symbol=loader_sym,
            start_date=(now - timedelta(days=5)).strftime("%Y-%m-%d"),
            end_date=now.strftime("%Y-%m-%d"),
            interval="60m",
            data_source="local",
        )
        logger.info(f"Loaded {len(df)} bars from local data")

    if len(df) < 30:
        logger.warning(f"Insufficient data ({len(df)} bars); need ≥30")
        return

    # 2) Feature engineering
    try:
        fs = fe.engineer_features(
            df, symbol=SYMBOL.replace("_", ""), additional_data={}
        )
        X = fs.features.drop(
            [c for c in fs.features.columns if c.startswith("future_")], axis=1
        )
        logger.info(f"Engineered features with shape {X.shape}")
    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        return

    # 3) Align X to the model’s training columns
    if hasattr(declared_model, "feature_names_in_"):
        trained_cols = list(declared_model.feature_names_in_)
        # zero-fill any missing
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        # drop extras & reorder
        X = X.reindex(columns=trained_cols)
        logger.info(f"Aligned features to trained set, new shape {X.shape}")
    for col in ["open","high","low","close","volume"]:
        X[col] = df[col].reindex(X.index)
    # 4) Generate signal
    try:
        sig_df = TradingStrategy(declared_model).generate_signals(X)
        last_idx = sig_df.index[-1]
        sig = int(sig_df["signal"].iloc[-1])
        price = float(df["close"].iloc[-1])
        logger.info(f"Signal at {last_idx}: {sig} @ {price}")
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return

    # 5) Skip duplicates
    if last_traded_time == last_idx or (sig == last_traded_signal and sig != 0):
        logger.info("No new signal; skipping trade")
        return

    # 6) Execute trade
    if sig != 0:
        try:
            units = 1000
            if LIVE_MODE:
                resp = OandaExecutor().place_order(
                    signal=sig, symbol=SYMBOL, units=units, price=price
                )
                logger.info(f"Live trade executed: {resp}")
            else:
                trades = Executor().execute(
                    pd.DataFrame({"signal": [sig]}, index=[last_idx]),
                    df[["close"]].iloc[-1:],
                )
                logger.info(f"Simulated trade: {trades}")
            last_traded_signal = sig
            last_traded_time = last_idx
        except Exception as e:
            logger.error(f"Trade execution error: {e}")





def setup_schedule():
    schedule.every().hour.at(":01").do(run_once)
    run_once()
    logger.info("Scheduler initialized; will run at :01 every hour")

if __name__ == "__main__":
    logger.info("🚀 Starting live_trade system")
    setup_schedule()
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("👋 live_trade stopped by user")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(60)
