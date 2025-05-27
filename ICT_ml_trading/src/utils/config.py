# src/utils/config.py

"""
Global configuration for the FX ML trading system.
"""

# === Risk & TP/SL Settings ===
USE_TP_SL        = False
TAKE_PROFIT_PIPS = 10
STOP_LOSS_PIPS   = 5

# === Pip Size Definitions ===
PIP_SIZE_DICT = {
    "DEFAULT": 0.0001,   # standard FX pairs
    "USD_JPY":  0.01,     # JPY crosses
    "XAU_USD": 0.01,     # gold
    "XAG_USD": 0.001,    # silver
    "US30_USD": 1.0,     # Dow Jones index
    "NAS100_USD": 1.0,   # Nasdaq-100 index
}
DEFAULT_PIP_SIZE = PIP_SIZE_DICT["DEFAULT"]


# === Brokers for copy-trading ===
# List every broker account you want to fan out trades to.
BROKERS = ["OANDA", "PEPPERSTONE", "FTMO", "FXCM"] # e.g. ["OANDA", "FXCM", "FTMO"]. you can add more. you have to have an executor class for each new broker you add

# === Instruments to trade ===
SYMBOLS = ["EUR_USD", "GBP_USD", "XAU_USD", "US30_USD", "NAS100_USD"] #you can add more

# === OANDA Credentials ===
OANDA_API_TOKEN  = "cefb66e4bf361e828b27a7e1875e9157-c2550c70ce264031c57e3ea268af107d"
OANDA_ACCOUNT_ID = "101-001-31678593-001"
OANDA_ENV        = "practice"   # "practice" or "live"

# === FXCM Credentials (if you add FXCM to BROKERS) ===
FXCM_API_TOKEN   = "YOUR_FXCM_TOKEN"

# === FTMO / MT5 Settings (if you add FTMO to BROKERS) === C:\Program Files\FTMO Global Markets MT5 Terminal
FTMO_MT5_TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
FTMO_MT5_LOGIN    = 1510802672
FTMO_MT5_PASSWORD = "n*S!5ELYH3p67"
FTMO_MT5_SERVER   = "FTMO-Demo"

# === Pepperstone / MT5 Settings (optional) ===
PEPPERSTONE_MT5_TERMINAL = r"C:\Pepperstone MetaTrader 5\terminal64.exe"
PEPPERSTONE_MT5_LOGIN    = 61361210          # put real login when ready
PEPPERSTONE_MT5_PASSWORD = "d-*9garFTpeyfP2"
PEPPERSTONE_MT5_SERVER   = "Pepperstone-Demo"
