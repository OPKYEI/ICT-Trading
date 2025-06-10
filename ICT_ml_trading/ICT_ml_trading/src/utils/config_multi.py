# src/utils/config.py
"""
Configuration and settings for the trading system.
"""

# ──────────────────────────────────────────────────────────
# 1) Broker list
# ──────────────────────────────────────────────────────────
BROKERS = ["OANDA", "FTMO", "PEPPERSTONE", "FXCM"] #add more as you wish

# ──────────────────────────────────────────────────────────
# 2) Instrument list
# ──────────────────────────────────────────────────────────
SYMBOLS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "XAU_USD",
    "US30_USD",
    "NAS100_USD"
] #add more as you wish

# ──────────────────────────────────────────────────────────
# 3) Pip size configuration
# ──────────────────────────────────────────────────────────
PIP_SIZE_DICT = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "NZD_USD": 0.0001,
    "USD_CAD": 0.0001,
    "USD_CHF": 0.0001,
    "XAU_USD": 0.01,
    "US30_USD": 1.0,
    "NAS100_USD": 1.0,
}
DEFAULT_PIP_SIZE = 0.0001

# ──────────────────────────────────────────────────────────
# 4) TP/SL Configuration
# ──────────────────────────────────────────────────────────
USE_TP_SL = False
TAKE_PROFIT_PIPS = 50
STOP_LOSS_PIPS = 30

# ──────────────────────────────────────────────────────────
# 5) Multi-Account Broker Configuration
# ──────────────────────────────────────────────────────────
# This structure allows multiple accounts per broker
# Each account has a unique identifier (e.g., "FTMO_1", "FTMO_2")

BROKER_ACCOUNTS = {
    # OANDA accounts
    "OANDA_1": {
        "broker_type": "OANDA",
        "api_token": "cefb66e4bf361e828b27a7e1875e9157-c2550c70ce264031c57e3ea268af107d",
        "account_id": "101-001-31678593-001",
        "environment": "practice",  # or "live"
        "enabled": True,
        "symbols": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"] #oanda US doesn't allow CFDs. OANDA europe, australia, and elsewhere does. so you can configure for the CFDs if you are not using OANDA USA
    },
    
    # FTMO accounts
    "FTMO_1": {
        "broker_type": "FTMO",
        "terminal": r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
        "login": 1510802672,
        "password": "n*S!5ELYH3p67",
        "server": "FTMO-Demo",
        "enabled": True,
        "symbols": SYMBOLS,  # All symbols
        "magic": 234000
    },
    "FTMO_2": {
        "broker_type": "FTMO",
        "terminal": r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
        "login": 1510826805,
        "password": "!*X66$Gi*jbM8",
        "server": "FTMO-Demo",
        "enabled": True,
        "symbols": SYMBOLS,  # All symbols
        "magic": 234001
    },
    
    # Pepperstone accounts
    "PEPPERSTONE_1": {
        "broker_type": "PEPPERSTONE",
        "terminal": r"C:\Pepperstone MetaTrader 5\terminal64.exe",
        "login": 61361210,  # Your login here
        "password": "d-*9garFTpeyfP2",
        "server": "Pepperstone-Demo",
        "enabled": True,
        "symbols": SYMBOLS,
        "magic": 234100
    },
    
    # FXCM accounts
    "FXCM_1": {
        "broker_type": "FXCM",
        "api_token": "YOUR_FXCM_TOKEN",
        "enabled": False,
        "symbols": SYMBOLS
    }
}

# ──────────────────────────────────────────────────────────
# 6) Legacy single-account configuration (for backward compatibility)
# ──────────────────────────────────────────────────────────
# These are kept for backward compatibility but won't be used if BROKER_ACCOUNTS is defined

# OANDA settings
OANDA_API_TOKEN = "cefb66e4bf361e828b27a7e1875e9157-c2550c70ce264031c57e3ea268af107d"
OANDA_ACCOUNT_ID = "101-001-31678593-001"
OANDA_ENV = "practice"

# FTMO settings
FTMO_MT5_TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
FTMO_MT5_LOGIN = 0
FTMO_MT5_PASSWORD = "n*S!5ELYH3p67"
FTMO_MT5_SERVER = "FTMO-Demo"

# Pepperstone settings
PEPPERSTONE_MT5_TERMINAL = r"C:\Pepperstone MetaTrader 5\terminal64.exe"
PEPPERSTONE_MT5_LOGIN = 0
PEPPERSTONE_MT5_PASSWORD = "d-*9garFTpeyfP2"
PEPPERSTONE_MT5_SERVER = "Pepperstone-Demo"

# FXCM settings
FXCM_API_TOKEN = "YOUR_FXCM_TOKEN"