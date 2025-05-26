#!/usr/bin/env python3
# live_trade.py

import sys, os, ctypes, logging, warnings, time, schedule
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path("src").resolve()))


# ──────────────────────────────────────────────────────────
# 1) UTF-8 console on Windows
# ──────────────────────────────────────────────────────────
def enable_unicode_console():
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
        except:
            pass
enable_unicode_console()

# ──────────────────────────────────────────────────────────
# 2) Logging setup
# ──────────────────────────────────────────────────────────
log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
logger = logging.getLogger("live_trade")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
for h in [logging.StreamHandler(sys.stdout),
          logging.FileHandler(log_dir/"live_trade.log", encoding="utf-8")]:
    h.setFormatter(formatter); h.setLevel(logging.INFO); logger.addHandler(h)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 3) Imports & Configuration
# ──────────────────────────────────────────────────────────
import joblib
import pandas as pd
from src.utils.config import (
    BROKERS, SYMBOLS,
    PIP_SIZE_DICT, DEFAULT_PIP_SIZE,
    OANDA_API_TOKEN, OANDA_ACCOUNT_ID, OANDA_ENV,
    FXCM_API_TOKEN,
    FTMO_MT5_TERMINAL, FTMO_MT5_LOGIN, FTMO_MT5_PASSWORD, FTMO_MT5_SERVER
)
from src.data_processing.data_loader import DataLoader
from src.features.feature_engineering import ICTFeatureEngineer
from src.trading.strategy import TradingStrategy
from src.trading.executor import Executor, OandaExecutor, FTMOExecutor
# (you can import FXCM/FTMO executors here once implemented)

# ──────────────────────────────────────────────────────────
# 4) Determine live vs simulation
# ──────────────────────────────────────────────────────────
LIVE_MODE = ("OANDA" in BROKERS and OANDA_API_TOKEN != "YOUR_OANDA_TOKEN")
mode = "LIVE" if LIVE_MODE else "SIMULATION"
logger.info(f"Starting in {mode} mode")

# ──────────────────────────────────────────────────────────
# 5) Initialize data loader & feature engineer
# ──────────────────────────────────────────────────────────
data_loader = DataLoader(data_path=Path("data"))
fe = ICTFeatureEngineer(lookback_periods=[5,10,20], feature_selection_threshold=0.01)

# ──────────────────────────────────────────────────────────
# 6) Build signal‐provider (OANDA or local CSV)
# ──────────────────────────────────────────────────────────
try:
    from src.data_processing.oanda_data import OandaDataFetcher
    signal_fetcher = OandaDataFetcher() if LIVE_MODE else None
    if signal_fetcher:
        logger.info("Initialized OandaDataFetcher for signals")
except ImportError:
    logger.warning("OandaDataFetcher not available; falling back to local CSV for signals")
    signal_fetcher = None

# ──────────────────────────────────────────────────────────
# 7) Instantiate every broker‐executor
# ──────────────────────────────────────────────────────────
executors = {}
for broker in BROKERS:
    if broker == "OANDA":
        executors[broker] = OandaExecutor()
    elif broker == "FTMO":
        executors[broker] = FTMOExecutor(
            FTMO_MT5_TERMINAL,
            FTMO_MT5_LOGIN,
            FTMO_MT5_PASSWORD,
            FTMO_MT5_SERVER
        )
    else:
        executors[broker] = Executor()

logger.info(f"Registered executors for brokers: {list(executors)}")

# ──────────────────────────────────────────────────────────
# 8) Load pipeline model once
# ──────────────────────────────────────────────────────────
def to_numpy_array(X): return X.values

def load_model():
    # choose the latest '*_best_pipeline_*.pkl' in checkpoints/
    ckpt = sorted((Path("checkpoints")\
            .glob("*_best_pipeline_*.pkl")))[-1]
    logger.info(f"Loading model: {ckpt.name}")
    mdl = joblib.load(ckpt)
    return mdl

model = load_model()

# ──────────────────────────────────────────────────────────
# 9) Track last trades to avoid repetition (per symbol)
# ──────────────────────────────────────────────────────────
last_traded_signal = {sym:0 for sym in SYMBOLS}
last_traded_time   = {sym:None for sym in SYMBOLS}

# ──────────────────────────────────────────────────────────
# 10) The trading cycle
# ──────────────────────────────────────────────────────────
def run_once():
    now = datetime.now()
    logger.info("▶️ Running trading cycle")

    # For each symbol, fetch data once and generate a signal
    for symbol in SYMBOLS:
        # 1) Fetch OHLC data
        df = None
        if signal_fetcher:
            try:
                instr = symbol if "_" in symbol else f"{symbol[:3]}_{symbol[3:]}"
                df = signal_fetcher.fetch_ohlc(
                    instrument=instr,
                    granularity="H1",
                    count=120
                )
                logger.info(f"{symbol}: fetched {len(df)} bars from Oanda")
            except Exception as e:
                logger.warning(f"{symbol}: Oanda fetch failed: {e}")

        if df is None or df.empty:
            df = data_loader.load_data(
                symbol.replace("_", ""),
                start_date=(now - timedelta(days=5)).strftime("%Y-%m-%d"),
                end_date=now.strftime("%Y-%m-%d"),
                interval="60m",
                data_source="local"
            )
            logger.info(f"{symbol}: loaded {len(df)} bars from local CSV")

        if len(df) < 30:
            logger.warning(f"{symbol}: Insufficient data ({len(df)} bars)—skipping")
            continue

        # 2) Feature engineering
        try:
            fs = fe.engineer_features(
                data=df,
                symbol=symbol.replace("_", ""),
                additional_data={}
            )
            X = fs.features.drop(
                [c for c in fs.features.columns if c.startswith("future_")],
                axis=1
            )
            logger.info(f"{symbol}: engineered features shape {X.shape}")
        except Exception as e:
            logger.error(f"{symbol}: feature-engineering error: {e}")
            continue

        # 3) Align features & re-attach raw OHLCV for strategy logic
        if hasattr(model, "feature_names_in_"):
            cols = model.feature_names_in_
            for c in cols:
                if c not in X:
                    X[c] = 0
            X = X.reindex(columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            X[c] = df[c].reindex(X.index)

        # 4) Generate a signal
        try:
            sig_df = TradingStrategy(model).generate_signals(X)
            last_idx = sig_df.index[-1]
            sig      = int(sig_df["signal"].iloc[-1])
            price    = float(df["close"].iloc[-1])
            logger.info(f"{symbol}: signal {sig} @ {price}")
        except Exception as e:
            logger.error(f"{symbol}: signal generation error: {e}")
            continue

        # 5) Skip duplicates
        if last_traded_time[symbol] == last_idx or (sig == last_traded_signal[symbol] and sig != 0):
            logger.info(f"{symbol}: no new signal—skipping")
            continue

        # 6) Broadcast the trade to ALL brokers
        if sig != 0:
            for broker, executor_instance in executors.items():
                try:
                    units = 1000  # adjust per symbol or broker if needed

                    # Live brokers use place_order()
                    if LIVE_MODE and hasattr(executor_instance, "place_order"):
                        resp = executor_instance.place_order(
                            signal=sig,
                            symbol=symbol,
                            units=units,
                            price=price
                        )
                        logger.info(f"{broker} trade executed for {symbol}: {resp}")
                    # Fallback simulation or non‐live executors
                    elif hasattr(executor_instance, "execute"):
                        trades = executor_instance.execute(
                            pd.DataFrame({"signal": [sig]}, index=[last_idx]),
                            df[["close"]].iloc[-1:]
                        )
                        logger.info(f"{broker} simulated trade for {symbol}: {trades}")
                    else:
                        logger.warning(f"{broker} has no execution method—skipping")
                except Exception as e:
                    logger.error(f"{broker} execution error for {symbol}: {e}")

            last_traded_signal[symbol] = sig
            last_traded_time[symbol]   = last_idx


# ──────────────────────────────────────────────────────────
# 11) Scheduler
# ──────────────────────────────────────────────────────────
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
