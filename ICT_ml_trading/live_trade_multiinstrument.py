#!/usr/bin/env python3
# live_trade_multiinstrument.py

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
from src.trading.executor import Executor, OandaExecutor, FTMOExecutor, PepperstoneExecutor

# Try to import Pepperstone if available
try:
    from src.utils.config import (
        PEPPERSTONE_MT5_TERMINAL, PEPPERSTONE_MT5_LOGIN, 
        PEPPERSTONE_MT5_PASSWORD, PEPPERSTONE_MT5_SERVER
    )
    from src.trading.executor import PepperstoneExecutor
    PEPPERSTONE_AVAILABLE = True
except ImportError:
    PEPPERSTONE_AVAILABLE = False
    logger.debug("Pepperstone configuration not found")

# ──────────────────────────────────────────────────────────
# 4) Helper functions
# ──────────────────────────────────────────────────────────
def clean_duplicate_timestamps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Remove duplicate timestamps from dataframe"""
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.warning(f"{symbol}: Found {dup_count} duplicate timestamps, keeping last")
        df = df[~df.index.duplicated(keep='last')]
    return df

# ──────────────────────────────────────────────────────────
# 5) Determine if we have any live brokers configured
# ──────────────────────────────────────────────────────────
CONFIGURED_BROKERS = {
    "OANDA": OANDA_API_TOKEN != "YOUR_OANDA_TOKEN",
    "FTMO": FTMO_MT5_LOGIN != 0,
    "FXCM": FXCM_API_TOKEN != "YOUR_FXCM_TOKEN"
}

# Add Pepperstone if available
if PEPPERSTONE_AVAILABLE:
    CONFIGURED_BROKERS["PEPPERSTONE"] = PEPPERSTONE_MT5_LOGIN != 0

# Check which brokers are actually configured
ACTIVE_BROKERS = {name: True for name, configured in CONFIGURED_BROKERS.items() 
                  if name in BROKERS and configured}

logger.info(f"Configured brokers: {list(ACTIVE_BROKERS.keys())}")
logger.info(f"Mode: LIVE TRADING on {len(ACTIVE_BROKERS)} broker(s)")

# ──────────────────────────────────────────────────────────
# 6) Initialize data loader & feature engineer
# ──────────────────────────────────────────────────────────
data_loader = DataLoader(data_path=Path("data"))
fe = ICTFeatureEngineer(lookback_periods=[5,10,20], feature_selection_threshold=0.01)

# ──────────────────────────────────────────────────────────
# 7) Build signal‐provider (prefer live data when available)
# ──────────────────────────────────────────────────────────
signal_fetcher = None
if "OANDA" in ACTIVE_BROKERS:
    try:
        from src.data_processing.oanda_data import OandaDataFetcher
        signal_fetcher = OandaDataFetcher()
        logger.info("📡 Using OandaDataFetcher for live market data")
    except ImportError:
        logger.warning("⚠️ OandaDataFetcher not available")

if not signal_fetcher:
    logger.info("📁 Using local CSV data for signals")

# ──────────────────────────────────────────────────────────
# 8) Instantiate every broker‐executor
# ──────────────────────────────────────────────────────────
executors = {}
broker_instruments = {}  # Track which instruments each broker supports

for broker in BROKERS:
    try:
        if broker == "OANDA" and CONFIGURED_BROKERS.get("OANDA", False):
            executors[broker] = OandaExecutor()
            # OANDA demo supports forex and some commodities
            broker_instruments[broker] = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "XAU_USD", "XAG_USD"]
            logger.info(f"✅ Initialized {broker} executor")
            
        elif broker == "FTMO" and CONFIGURED_BROKERS.get("FTMO", False):
            executors[broker] = FTMOExecutor(
                FTMO_MT5_TERMINAL,
                FTMO_MT5_LOGIN, 
                FTMO_MT5_PASSWORD,
                FTMO_MT5_SERVER
            )
            # FTMO supports all instruments
            broker_instruments[broker] = SYMBOLS.copy()
            logger.info(f"✅ Initialized {broker} executor")
            
        elif broker == "PEPPERSTONE" and CONFIGURED_BROKERS.get("PEPPERSTONE", False) and PEPPERSTONE_AVAILABLE:
            executors[broker] = PepperstoneExecutor(
                PEPPERSTONE_MT5_TERMINAL,
                PEPPERSTONE_MT5_LOGIN,
                PEPPERSTONE_MT5_PASSWORD,
                PEPPERSTONE_MT5_SERVER
            )
            # Pepperstone supports all instruments including indices
            broker_instruments[broker] = SYMBOLS.copy()
            logger.info(f"✅ Initialized {broker} executor")
            
        else:
            logger.warning(f"⚠️ {broker} is in BROKERS list but not configured or not implemented")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize {broker}: {e}")
        logger.info(f"   → Continuing without {broker}")

if not executors:
    logger.error("❌ No brokers successfully initialized! Check your configuration.")
    logger.info("   → Set up at least one broker's credentials in config.py")
else:
    logger.info(f"📊 Active brokers: {list(executors.keys())}")

# ──────────────────────────────────────────────────────────
# 9) Load pipeline model once
# ──────────────────────────────────────────────────────────
def to_numpy_array(X): return X.values

def load_model():
    # choose the latest '*_best_pipeline_*.pkl' in checkpoints/
    ckpt = sorted((Path("checkpoints").glob("*_best_pipeline_*.pkl")))[-1]
    logger.info(f"Loading model: {ckpt.name}")
    mdl = joblib.load(ckpt)
    return mdl

model = load_model()

# ──────────────────────────────────────────────────────────
# 10) Track last trades to avoid repetition (per symbol)
# ──────────────────────────────────────────────────────────
last_traded_signal = {sym:0 for sym in SYMBOLS}
last_traded_time   = {sym:None for sym in SYMBOLS}

# ──────────────────────────────────────────────────────────
# 11) The trading cycle
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

        # Clean any duplicate timestamps
        df = clean_duplicate_timestamps(df, symbol)

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
            if c in df.columns:
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

        # 6) Broadcast the trade to ALL configured brokers
        
        if sig != 0:
            successful_brokers = []
            failed_brokers = []
            
            for broker, executor_instance in executors.items():
                # Check if this broker supports this instrument
                supported_instruments = broker_instruments.get(broker, SYMBOLS)
                if symbol not in supported_instruments:
                    logger.info(f"📌 {broker}: {symbol} not supported, skipping")
                    continue
                    
                try:
                    units = 50000  # You can customize this per broker if needed
                    
                    # All brokers should have place_order for live trading
                    if hasattr(executor_instance, "place_order"):
                        resp = executor_instance.place_order(
                            signal=sig,
                            symbol=symbol,
                            units=units,
                            price=price
                        )
                        
                        # Check the response
                        if resp:
                            if isinstance(resp, dict):
                                # Check for OANDA responses
                                if "orderFillTransaction" in resp:
                                    logger.info(f"✅ {broker} executed {symbol} trade: order {resp['orderFillTransaction']['id']}")
                                    successful_brokers.append(broker)
                                elif "orderRejectTransaction" in resp:
                                    logger.error(f"❌ {broker} rejected {symbol} order: {resp['orderRejectTransaction']['rejectReason']}")
                                    failed_brokers.append(broker)
                                # Check for MT5 responses
                                elif "retcode" in resp:
                                    retcode = resp["retcode"]
                                    if retcode == 10009:  # TRADE_RETCODE_DONE
                                        order_id = resp.get("order", "N/A")
                                        logger.info(f"✅ {broker} executed {symbol} trade: order {order_id}, deal {resp.get('deal', 'N/A')}")
                                        successful_brokers.append(broker)
                                    else:
                                        comment = resp.get("comment", "Unknown error")
                                        logger.error(f"❌ {broker} failed {symbol} trade: {comment} (retcode: {retcode})")
                                        failed_brokers.append(broker)
                                else:
                                    logger.warning(f"⚠️ {broker} unexpected response for {symbol}")
                                    failed_brokers.append(broker)
                            else:
                                logger.warning(f"⚠️ {broker} non-dict response for {symbol}")
                                failed_brokers.append(broker)
                        else:
                            logger.error(f"❌ {broker} returned None for {symbol}")
                            failed_brokers.append(broker)
                    else:
                        logger.warning(f"⚠️ {broker} executor missing place_order() method")
                        failed_brokers.append(broker)
                        
                except Exception as e:
                    logger.error(f"❌ {broker} execution failed for {symbol}: {e}")
                    failed_brokers.append(broker)
            
            # Update tracking only if at least one broker succeeded
            if successful_brokers:
                last_traded_signal[symbol] = sig
                last_traded_time[symbol] = last_idx
                logger.info(f"📊 Trade summary for {symbol}: "
                          f"Success: {successful_brokers}, "
                          f"Failed: {failed_brokers}")

# ──────────────────────────────────────────────────────────
# 12) Scheduler
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