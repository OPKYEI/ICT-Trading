#!/usr/bin/env python3
# live_trade_multiinstrument_improved.py

import sys, os, ctypes, logging, warnings, time, schedule
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path("src").resolve()))

def to_numpy_array(X):
    """Convert pandas DataFrame to numpy array for the pipeline."""
    return X.values
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
# 4) Symbol-to-Model Mapping & Model Manager
# ──────────────────────────────────────────────────────────
class ModelManager:
    """Manages loading and caching of symbol-specific models"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.models = {}  # Cache loaded models
        self.symbol_mapping = self._create_symbol_mapping()
        
    def _create_symbol_mapping(self):
        """Map trading symbols to their CSV file names"""
        return {
            "GBP_USD": "GBPUSD=X_60m",
            "GBPUSD": "GBPUSD=X_60m",
            "EUR_USD": "EURUSD=X_60m", 
            "EURUSD": "EURUSD=X_60m",
            "USD_JPY": "USDJPY=X_60m",
            "USDJPY": "USDJPY=X_60m",
            "AUD_USD": "AUDUSD=X_60m",
            "AUDUSD": "AUDUSD=X_60m",
            "USD_CAD": "USDCAD=X_60m",
            "USDCAD": "USDCAD=X_60m",
            "XAU_USD": "XAUUSD=X_60m",
            "XAUUSD": "XAUUSD=X_60m",
            "XAG_USD": "XAGUSD=X_60m",
            "XAGUSD": "XAGUSD=X_60m",
            # Update these to match your actual model files
            "US30_USD": "USA30=X_60m",    # Changed to match your file
            "US30USD": "USA30=X_60m",
            "NAS100_USD": "USATECH=X_60m", # Changed to match your file
            "NAS100USD": "USATECH=X_60m",
        }
    
    def _find_best_model_file(self, csv_name):
        """Find the best model file for a given CSV name"""
        # Look for files matching the pattern: {csv_name}_best_pipeline_*.pkl
        pattern = f"{csv_name}_best_pipeline_*.pkl"
        matches = list(self.checkpoint_dir.glob(pattern))
        
        if matches:
            # If multiple matches, take the most recent one
            latest_file = max(matches, key=lambda x: x.stat().st_mtime)
            return latest_file
        
        # Fallback: look for any pipeline file for this symbol
        pattern = f"{csv_name}_*_pipeline.pkl"
        matches = list(self.checkpoint_dir.glob(pattern))
        
        if matches:
            latest_file = max(matches, key=lambda x: x.stat().st_mtime)
            return latest_file
            
        return None
    
    def get_model(self, symbol):
        """Get model for a specific symbol, loading if necessary"""
        # Normalize symbol (remove underscores for lookup)
        normalized_symbol = symbol.replace("_", "")
        
        # Check if already loaded
        if symbol in self.models:
            return self.models[symbol]
        
        # Get CSV name for this symbol
        csv_name = self.symbol_mapping.get(symbol) or self.symbol_mapping.get(normalized_symbol)
        
        if not csv_name:
            logger.warning(f"No model mapping found for symbol {symbol}")
            return None
        
        # Find the model file
        model_file = self._find_best_model_file(csv_name)
        
        if not model_file:
            logger.warning(f"No model file found for {symbol} (looking for {csv_name})")
            return None
        
        # Load the model
        try:
            model = joblib.load(model_file)
            self.models[symbol] = model
            logger.info(f"✅ Loaded model for {symbol}: {model_file.name}")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model for {symbol} from {model_file}: {e}")
            return None
    
    def preload_all_models(self, symbols):
        """Preload models for all symbols to catch issues early"""
        logger.info(f"🔄 Preloading models for {len(symbols)} symbols...")
        
        success_count = 0
        for symbol in symbols:
            model = self.get_model(symbol)
            if model is not None:
                success_count += 1
            else:
                logger.warning(f"⚠️ Could not load model for {symbol}")
        
        logger.info(f"✅ Successfully loaded {success_count}/{len(symbols)} models")
        return success_count > 0

# ──────────────────────────────────────────────────────────
# 5) Trade State Management (same as before)
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
                'active_trades': {},  # {symbol: {signal, entry_time, entry_price, position_tickets}}
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
    
    def add_trade(self, symbol, signal, entry_time, entry_price, position_tickets=None):
        """Record a new trade with position tickets"""
        if 'active_trades' not in self.state:
            self.state['active_trades'] = {}
            
        self.state['active_trades'][symbol] = {
            'signal': signal,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'position_tickets': position_tickets or {}  # {broker: ticket}
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
        
        # Ensure entry_time is a Timestamp object
        entry_time = trade['entry_time']
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
            
        hours_elapsed = (current_time - entry_time).total_seconds() / 3600
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
            # Ensure entry_time is a Timestamp object
            entry_time = trade['entry_time']
            if isinstance(entry_time, str):
                entry_time = pd.Timestamp(entry_time)
                
            hours_elapsed = (current_time - entry_time).total_seconds() / 3600
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
# 6) Helper functions (same as before)
# ──────────────────────────────────────────────────────────
def clean_duplicate_timestamps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Remove duplicate timestamps from dataframe"""
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.warning(f"{symbol}: Found {dup_count} duplicate timestamps, keeping last")
        df = df[~df.index.duplicated(keep='last')]
    return df

def close_positions_properly(symbol, trade_info, executors, broker_instruments):
    """Close positions using proper methods for each broker type"""
    
    position_tickets = trade_info.get('position_tickets', {})
    
    for broker, executor in executors.items():
        supported = broker_instruments.get(broker, SYMBOLS)
        if symbol not in supported:
            continue
            
        try:
            if broker == "OANDA":
                # OANDA doesn't support hedging - use opposite signal
                close_signal = -trade_info['signal']
                resp = executor.place_order(
                    signal=close_signal,
                    symbol=symbol,
                    units=10000,
                    price=None
                )
                if resp:
                    logger.info(f"✅ {broker} closed {symbol} position")
                    
            elif broker in ["FTMO", "PEPPERSTONE"]:
                # MT5 brokers - use position-specific closing
                ticket = position_tickets.get(broker)
                
                if ticket:
                    # Try to close specific position
                    try:
                        result = executor.close_position(ticket)
                        if result and result.get('retcode') == 10009:
                            logger.info(f"✅ {broker} closed {symbol} position {ticket}")
                        else:
                            logger.error(f"❌ {broker} failed to close position {ticket}: {result}")
                    except Exception as e:
                        logger.warning(f"⚠️ {broker} position {ticket} not found, may be already closed: {e}")
                else:
                    # No ticket stored - close all positions for this symbol with our magic
                    magic = 234000 if broker == "FTMO" else 234001
                    closed = executor.close_positions_by_symbol_and_magic(symbol, magic)
                    if closed:
                        logger.info(f"✅ {broker} closed {len(closed)} {symbol} positions")
                    else:
                        logger.warning(f"⚠️ {broker} no positions found for {symbol}")
                        
        except Exception as e:
            logger.error(f"❌ {broker} failed to close {symbol}: {e}")

def check_and_close_expired_trades(state_manager, executors, broker_instruments):
    """Check all active trades and close any that have exceeded 5 hours"""
    current_time = pd.Timestamp.now()
    
    for symbol, trade_info in list(state_manager.state.get('active_trades', {}).items()):
        entry_time = trade_info['entry_time']
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
            
        hours_elapsed = (current_time - entry_time).total_seconds() / 3600
        
        if hours_elapsed >= 5:
            logger.info(f"{symbol}: Trade exceeded 5-hour window ({hours_elapsed:.1f}h), closing position")
            
            # Use proper closing function
            close_positions_properly(symbol, trade_info, executors, broker_instruments)
            
            # Remove from state
            state_manager.remove_trade(symbol)

def sync_with_broker_positions(state_manager, executors):
    """Sync internal state with actual broker positions"""
    logger.info("Syncing with broker positions...")
    
    # This is a simplified version - you'd need to implement actual position checking
    # For now, we'll trust our state file
    pass

# ──────────────────────────────────────────────────────────
# 7) Determine if we have any live brokers configured
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
# 8) Initialize components
# ──────────────────────────────────────────────────────────
# Initialize state manager
state_manager = TradeStateManager()

# Initialize MODEL MANAGER (NEW)
model_manager = ModelManager()

# Initialize data loader & feature engineer
data_loader = DataLoader(data_path=Path("data"))
fe = ICTFeatureEngineer(lookback_periods=[5,10,20], feature_selection_threshold=0.01)

# ──────────────────────────────────────────────────────────
# 9) Build signal‐provider (prefer live data when available)
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
# 10) Instantiate every broker‐executor
# ──────────────────────────────────────────────────────────
executors = {}
broker_instruments = {}  # Track which instruments each broker supports

for broker in BROKERS:
    try:
        if broker == "OANDA" and CONFIGURED_BROKERS.get("OANDA", False):
            executors[broker] = OandaExecutor()
            broker_instruments[broker] = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
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
# 11) Preload all models (NEW)
# ──────────────────────────────────────────────────────────
logger.info("🤖 Preloading models for all symbols...")
if not model_manager.preload_all_models(SYMBOLS):
    logger.error("❌ No models could be loaded! Check your checkpoints directory.")
    logger.info("   → Make sure you've run the training pipeline for all symbols.")
else:
    logger.info(f"✅ Model loading completed")

# ──────────────────────────────────────────────────────────
# 12) The improved trading cycle (UPDATED)
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
        # 1) Get symbol-specific model (NEW)
        model = model_manager.get_model(symbol)
        if model is None:
            logger.warning(f"{symbol}: No model available, skipping")
            continue
        
        # 2) Fetch OHLC data
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

        # 3) Feature engineering
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

        # 4) Align features & re-attach raw OHLCV for strategy logic
        if hasattr(model, "feature_names_in_"):
            cols = model.feature_names_in_
            for c in cols:
                if c not in X:
                    X[c] = 0
            X = X.reindex(columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                X[c] = df[c].reindex(X.index)

        # 5) Generate a signal using symbol-specific model (UPDATED)
        try:
            sig_df = TradingStrategy(model).generate_signals(X)
            last_idx = sig_df.index[-1]
            sig = int(sig_df["signal"].iloc[-1])
            price = float(df["close"].iloc[-1])
            logger.info(f"{symbol}: signal {sig} @ {price} (model: {model_manager.symbol_mapping.get(symbol, 'unknown')})")
        except Exception as e:
            logger.error(f"{symbol}: signal generation error: {e}")
            continue

        # 6) Check if we should enter based on 5-bar window logic
        if sig == 0:
            logger.info(f"{symbol}: Flat signal, skipping")
            continue
            
        should_enter = state_manager.should_enter_trade(symbol, sig, price, current_time)
        
        if not should_enter:
            continue
            
        # 7) For late entries, check if price is still valid
        active_trade = state_manager.get_active_trade(symbol)
        if active_trade and not state_manager.is_trade_expired(symbol, current_time):
            # We're within 5-hour window but different direction or restarted
            if not state_manager.is_price_valid_for_late_entry(symbol, sig, price):
                logger.warning(f"{symbol}: Price moved too far from initial entry, skipping late entry")
                continue

        # 8) Close existing position if different direction
        if active_trade and active_trade['signal'] != sig:
            logger.info(f"{symbol}: Closing existing position before opening opposite")
            
            # Use proper closing function
            close_positions_properly(symbol, active_trade, executors, broker_instruments)

        # 9) Execute new trade on all brokers
        successful_brokers = []
        failed_brokers = []
        position_tickets = {}  # Store position tickets for each broker
        
        for broker, executor_instance in executors.items():
            # Check if this broker supports this instrument
            supported_instruments = broker_instruments.get(broker, SYMBOLS)
            if symbol not in supported_instruments:
                logger.info(f"📌 {broker}: {symbol} not supported, skipping")
                continue
                
            try:
                units = 10000
                
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
                                # OANDA doesn't use position tickets the same way
                                
                            elif "orderRejectTransaction" in resp:
                                logger.error(f"❌ {broker} rejected {symbol} order: {resp['orderRejectTransaction']['rejectReason']}")
                                failed_brokers.append(broker)
                                
                            # Check for MT5 responses
                            elif "retcode" in resp:
                                retcode = resp["retcode"]
                                if retcode == 10009:  # TRADE_RETCODE_DONE
                                    order_id = resp.get("order", "N/A")
                                    
                                    # Extract position ticket if available
                                    if "position_ticket" in resp:
                                        position_tickets[broker] = resp["position_ticket"]
                                        logger.info(f"✅ {broker} executed {symbol} trade: order {order_id}, position {resp['position_ticket']}")
                                    else:
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
        # Update state if at least one broker succeeded
        if successful_brokers:
            state_manager.add_trade(symbol, sig, current_time, price, position_tickets)
            logger.info(f"📊 Trade summary for {symbol}: "
                      f"Success: {successful_brokers}, "
                      f"Failed: {failed_brokers}")

# ──────────────────────────────────────────────────────────
# 13) Scheduler
# ──────────────────────────────────────────────────────────
def setup_schedule():
    schedule.every().hour.at(":01").do(run_once)
    run_once()
    logger.info("Scheduler initialized; will run at :01 every hour")

if __name__ == "__main__":
    logger.info("🚀 Starting improved live_trade system with symbol-specific models")
    logger.info("📋 Features:")
    logger.info("   - Symbol-specific model loading")
    logger.info("   - Trades valid for exactly 5 bars")
    logger.info("   - Automatic position closing after 5 hours")
    logger.info("   - State persistence across restarts")
    logger.info("   - Price validation for late entries")
    logger.info("   - Proper hedging broker support")
    
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