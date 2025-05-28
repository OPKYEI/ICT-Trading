#!/usr/bin/env python3
# live_trade_multiinstrument_improved.py

import sys, os, ctypes, logging, warnings, time, schedule
from pathlib import Path
from datetime import datetime, timedelta
import json

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
import MetaTrader5 as mt5
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
# 4) Trade State Management
# ──────────────────────────────────────────────────────────
class TradeStateManager:
    """Manages trade state persistence and 5-bar window logic"""
    
    def __init__(self, state_file="trade_state.json"):
        self.state_file = state_file
        self.state = self.load_state()
        
    def load_state(self):
        """Load state from file or initialize empty"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                # Convert string timestamps back to datetime
                for symbol in state.get('active_trades', {}):
                    if 'entry_time' in state['active_trades'][symbol]:
                        state['active_trades'][symbol]['entry_time'] = pd.Timestamp(
                            state['active_trades'][symbol]['entry_time']
                        )
                return state
        except:
            return {
                'active_trades': {},  # {symbol: {signal, entry_time, entry_price}}
                'last_check': None
            }
    
    def save_state(self):
        """Save state to file"""
        # Convert timestamps to strings for JSON serialization
        state_to_save = self.state.copy()
        for symbol in state_to_save.get('active_trades', {}):
            if 'entry_time' in state_to_save['active_trades'][symbol]:
                state_to_save['active_trades'][symbol]['entry_time'] = str(
                    state_to_save['active_trades'][symbol]['entry_time']
                )
        
        with open(self.state_file, 'w') as f:
            json.dump(state_to_save, f, indent=2)
    
    def add_trade(self, symbol, signal, entry_time, entry_price):
        """Record a new trade"""
        if 'active_trades' not in self.state:
            self.state['active_trades'] = {}
            
        self.state['active_trades'][symbol] = {
            'signal': signal,
            'entry_time': entry_time,
            'entry_price': entry_price
        }
        self.save_state()
        
    def remove_trade(self, symbol):
        """Remove a trade from active trades"""
        if symbol in self.state.get('active_trades', {}):
            del self.state['active_trades'][symbol]
            self.save_state()
            
    def get_active_trade(self, symbol):
        """Get active trade for a symbol"""
        return self.state.get('active_trades', {}).get(symbol)
    
    def is_trade_expired(self, symbol, current_time):
        """Check if a trade has exceeded 5-bar window"""
        trade = self.get_active_trade(symbol)
        if not trade:
            return True
            
        hours_elapsed = (current_time - trade['entry_time']).total_seconds() / 3600
        return hours_elapsed >= 5
    
    def should_enter_trade(self, symbol, signal, current_price, current_time):
        """Determine if we should enter a new trade"""
        trade = self.get_active_trade(symbol)
        
        # No active trade - enter
        if not trade:
            logger.info(f"{symbol}: No active trade, can enter new position")
            return True
        
        # Check if current trade is expired (5+ hours old)
        if self.is_trade_expired(symbol, current_time):
            logger.info(f"{symbol}: Previous trade expired (>5 hours), can enter new position")
            return True
        
        # If within 5-hour window and same direction, skip
        if trade['signal'] == signal:
            hours_elapsed = (current_time - trade['entry_time']).total_seconds() / 3600
            logger.info(f"{symbol}: Same direction signal within 5-hour window ({hours_elapsed:.1f}h elapsed), skipping")
            return False
        
        # Different direction within window - allow (will close existing first)
        return True
    
    def is_price_valid_for_late_entry(self, symbol, signal, current_price):
        """Check if current price is valid for late entry"""
        trade = self.get_active_trade(symbol)
        if not trade:
            return True
            
        # For buy: current price should be at or below initial entry
        if signal == 1:  # Buy
            return current_price <= trade['entry_price']
        # For sell: current price should be at or above initial entry
        elif signal == -1:  # Sell
            return current_price >= trade['entry_price']
        
        return True

# ──────────────────────────────────────────────────────────
# 5) Helper functions
# ──────────────────────────────────────────────────────────
def clean_duplicate_timestamps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Remove duplicate timestamps from dataframe"""
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.warning(f"{symbol}: Found {dup_count} duplicate timestamps, keeping last")
        df = df[~df.index.duplicated(keep='last')]
    return df

def check_and_close_expired_trades(state_manager, executors, broker_instruments):
    """Check all active trades and close any that have exceeded 5 hours"""
    current_time = pd.Timestamp.now()
    
    for symbol, trade_info in list(state_manager.state.get('active_trades', {}).items()):
        if state_manager.is_trade_expired(symbol, current_time):
            logger.info(f"{symbol}: Trade exceeded 5-hour window, closing position")
            
            # Close on all brokers
            for broker, executor in executors.items():
                supported = broker_instruments.get(broker, SYMBOLS)
                if symbol not in supported:
                    continue
                    
                try:
                    # Send opposite signal to close
                    close_signal = -trade_info['signal']  # Opposite of original
                    
                    if hasattr(executor, "place_order"):
                        resp = executor.place_order(
                            signal=close_signal,
                            symbol=symbol,
                            units=50000,
                            price=None  # Current market price
                        )
                        logger.info(f"✅ {broker} closed {symbol} position after 5 hours")
                except Exception as e:
                    logger.error(f"❌ {broker} failed to close {symbol}: {e}")
            
            # Remove from state
            state_manager.remove_trade(symbol)

def sync_with_broker_positions(state_manager, executors):
    """Sync internal state with actual broker positions"""
    logger.info("Syncing with broker positions...")
    
    # This is a simplified version - you'd need to implement actual position checking
    # For now, we'll trust our state file
    pass

# ──────────────────────────────────────────────────────────
# 6) Determine if we have any live brokers configured
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
# 7) Initialize components
# ──────────────────────────────────────────────────────────
# Initialize state manager
state_manager = TradeStateManager()

# Initialize data loader & feature engineer
data_loader = DataLoader(data_path=Path("data"))
fe = ICTFeatureEngineer(lookback_periods=[5,10,20], feature_selection_threshold=0.01)

# ──────────────────────────────────────────────────────────
# 8) Build signal‐provider (prefer live data when available)
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
# 9) Instantiate every broker‐executor
# ──────────────────────────────────────────────────────────
executors = {}
broker_instruments = {}  # Track which instruments each broker supports

for broker in BROKERS:
    try:
        if broker == "OANDA" and CONFIGURED_BROKERS.get("OANDA", False):
            executors[broker] = OandaExecutor()
            broker_instruments[broker] = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
            logger.info(f"✅ Initialized {broker} executor")
            
        elif broker == "FTMO" and CONFIGURED_BROKERS.get("FTMO", False):
            executors[broker] = FTMOExecutor(
                FTMO_MT5_TERMINAL,
                FTMO_MT5_LOGIN, 
                FTMO_MT5_PASSWORD,
                FTMO_MT5_SERVER
            )
            broker_instruments[broker] = SYMBOLS.copy()
            logger.info(f"✅ Initialized {broker} executor")
            
        elif broker == "PEPPERSTONE" and CONFIGURED_BROKERS.get("PEPPERSTONE", False) and PEPPERSTONE_AVAILABLE:
            executors[broker] = PepperstoneExecutor(
                PEPPERSTONE_MT5_TERMINAL,
                PEPPERSTONE_MT5_LOGIN,
                PEPPERSTONE_MT5_PASSWORD,
                PEPPERSTONE_MT5_SERVER
            )
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
# 10) Load pipeline model once
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
# 11) The improved trading cycle
# ──────────────────────────────────────────────────────────
def run_once():
    current_time = pd.Timestamp.now()
    logger.info("▶️ Running trading cycle")
    
    # First, check and close any expired trades (>5 hours)
    check_and_close_expired_trades(state_manager, executors, broker_instruments)
    
    # Sync with broker positions on startup
    if state_manager.state.get('last_check') is None:
        sync_with_broker_positions(state_manager, executors)
    
    state_manager.state['last_check'] = str(current_time)
    state_manager.save_state()
    
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
                start_date=(current_time - timedelta(days=5)).strftime("%Y-%m-%d"),
                end_date=current_time.strftime("%Y-%m-%d"),
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
            sig = int(sig_df["signal"].iloc[-1])
            price = float(df["close"].iloc[-1])
            logger.info(f"{symbol}: signal {sig} @ {price}")
        except Exception as e:
            logger.error(f"{symbol}: signal generation error: {e}")
            continue

        # 5) Check if we should enter based on 5-bar window logic
        if sig == 0:
            logger.info(f"{symbol}: Flat signal, skipping")
            continue
            
        should_enter = state_manager.should_enter_trade(symbol, sig, price, current_time)
        
        if not should_enter:
            continue
            
        # 6) For late entries, check if price is still valid
        active_trade = state_manager.get_active_trade(symbol)
        if active_trade and not state_manager.is_trade_expired(symbol, current_time):
            # We're within 5-hour window but different direction or restarted
            if not state_manager.is_price_valid_for_late_entry(symbol, sig, price):
                logger.warning(f"{symbol}: Price moved too far from initial entry, skipping late entry")
                continue

        # 7) Close existing position if different direction
        if active_trade and active_trade['signal'] != sig:
            logger.info(f"{symbol}: Closing existing position before opening opposite")
            # Close existing position first
            for broker, executor in executors.items():
                supported = broker_instruments.get(broker, SYMBOLS)
                if symbol not in supported:
                    continue
                    
                try:
                    close_signal = -active_trade['signal']
                    if hasattr(executor, "place_order"):
                        resp = executor.place_order(
                            signal=close_signal,
                            symbol=symbol,
                            units=50000,
                            price=price
                        )
                        logger.info(f"✅ {broker} closed existing {symbol} position")
                except Exception as e:
                    logger.error(f"❌ {broker} failed to close {symbol}: {e}")

        # 8) Execute new trade on all brokers
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
                                    logger.info(f"✅ {broker} executed {symbol} trade: order {order_id}")
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
        
        # Update state if at least one broker succeeded
        if successful_brokers:
            state_manager.add_trade(symbol, sig, current_time, price)
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
    logger.info("🚀 Starting improved live_trade system with 5-bar window logic")
    logger.info("📋 Features:")
    logger.info("   - Trades valid for exactly 5 bars")
    logger.info("   - Automatic position closing after 5 hours")
    logger.info("   - State persistence across restarts")
    logger.info("   - Price validation for late entries")
    
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