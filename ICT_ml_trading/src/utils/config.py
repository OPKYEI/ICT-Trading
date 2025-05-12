# src/utils/config.py
"""
Global configuration for the FX ML trading system.
All user‑tweakable parameters live here. No need to edit any other file.
"""

# === Take-Profit / Stop-Loss Options ===
# When False: next-bar execution only (original behavior)
# When True: per-bar returns clipped to TP/SL levels (in pips)
USE_TP_SL        = False
TAKE_PROFIT_PIPS = 10    # in pips
STOP_LOSS_PIPS   = 5     # in pips

# === Pip Size Definitions ===
# Price units per pip by symbol
PIP_SIZE_DICT = {
    "DEFAULT": 0.0001,  # e.g. EURUSD, GBPUSD
    "USDJPY": 0.01,     # JPY pairs
}
DEFAULT_PIP_SIZE = PIP_SIZE_DICT["DEFAULT"]

# === Broker / Demo Trading Config ===
# FTMO has no public API; switch to OANDA or FXCM if needed
BROKER_NAME      = "OANDA"  # options: "FTMO", "OANDA", "FXCM"

# OANDA demo credentials (fallback)
OANDA_API_TOKEN  = "cefb66e4bf361e828b27a7e1875e9157-c2550c70ce264031c57e3ea268af107d"
OANDA_ACCOUNT_ID = "101-001-31678593-001"
OANDA_ENV = "practice" # 'practice' for simulation; 'live' for real trading

# FXCM demo credentials (fallback)
FXCM_API_TOKEN   = "YOUR_FXCM_TOKEN"

#===SYMBOL & INSTRUMENT===
SYMBOL = "EUR_USD"