# src/utils/config.py

"""
Global configuration for the FX ML trading system.
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in project root (2 levels up from src/utils/)
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not installed. Using environment variables only.")
    print("   Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# === Risk & TP/SL Settings ===
USE_TP_SL        = os.getenv('USE_TP_SL', 'False').lower() == 'true'
TAKE_PROFIT_PIPS = int(os.getenv('TAKE_PROFIT_PIPS', '10'))
STOP_LOSS_PIPS   = int(os.getenv('STOP_LOSS_PIPS', '5'))

# === Pip Size Definitions ===
PIP_SIZE_DICT = {
    "DEFAULT": 0.0001,   # standard FX pairs
    "USD_JPY":  0.01,     # JPY crosses
    "XAU_USD": 0.01,     # gold
    "GBP_JPY": 0.01,
    "XAG_USD": 0.001,    # silver
    "US30_USD": 1.0,     # Dow Jones index
    "NAS100_USD": 1.0,   # Nasdaq-100 index
}
DEFAULT_PIP_SIZE = PIP_SIZE_DICT["DEFAULT"]

# === Brokers for copy-trading ===
# List every broker account you want to fan out trades to.
BROKERS = ["OANDA", "PEPPERSTONE", "FTMO", "FXCM"] # you can add more

# === Instruments to trade ===
SYMBOLS = ["GBP_JPY", "USD_JPY", "EUR_USD", "USD_CAD", "US30_USD", "XAG_USD"] #add more as you desire

# === OANDA Credentials ===
OANDA_API_TOKEN  = os.getenv('OANDA_API_TOKEN', 'your_token_here')
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', 'your_account_id_here')
OANDA_ENV        = os.getenv('OANDA_ENV', 'practice')   # "practice" or "live". 'practice' -demo accounts; 'live' -actual live

# === FXCM Credentials ===
FXCM_API_TOKEN   = os.getenv('FXCM_API_TOKEN', 'your_fxcm_token_here')

# === FTMO / MT5 Settings ===
FTMO_MT5_TERMINAL = os.getenv('FTMO_MT5_TERMINAL', r'C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe') #(or whereever your directory path is)
FTMO_MT5_LOGIN    = int(os.getenv('FTMO_MT5_LOGIN', '0'))
FTMO_MT5_PASSWORD = os.getenv('FTMO_MT5_PASSWORD', 'your_password_here')
FTMO_MT5_SERVER   = os.getenv('FTMO_MT5_SERVER', 'FTMO-Demo')

# === Pepperstone / MT5 Settings ===
PEPPERSTONE_MT5_TERMINAL = os.getenv('PEPPERSTONE_MT5_TERMINAL', r'C:\Pepperstone MetaTrader 5\terminal64.exe') #(or whereever your directory path is)
PEPPERSTONE_MT5_LOGIN    = int(os.getenv('PEPPERSTONE_MT5_LOGIN', '0'))
PEPPERSTONE_MT5_PASSWORD = os.getenv('PEPPERSTONE_MT5_PASSWORD', 'your_password_here')
PEPPERSTONE_MT5_SERVER   = os.getenv('PEPPERSTONE_MT5_SERVER', 'Pepperstone-Demo')

# === Configuration Validation ===
def validate_config():
    """Validate that required credentials are set for active brokers"""
    missing_vars = []
    
    # Check OANDA credentials if in BROKERS list
    if "OANDA" in BROKERS:
        if OANDA_API_TOKEN == 'your_token_here':
            missing_vars.append('OANDA_API_TOKEN')
        if OANDA_ACCOUNT_ID == 'your_account_id_here':
            missing_vars.append('OANDA_ACCOUNT_ID')
    
    # Check FTMO credentials if in BROKERS list
    if "FTMO" in BROKERS:
        if FTMO_MT5_LOGIN == 0:
            missing_vars.append('FTMO_MT5_LOGIN')
        if FTMO_MT5_PASSWORD == 'your_password_here':
            missing_vars.append('FTMO_MT5_PASSWORD')
    
    # Check Pepperstone credentials if in BROKERS list
    if "PEPPERSTONE" in BROKERS:
        if PEPPERSTONE_MT5_LOGIN == 0:
            missing_vars.append('PEPPERSTONE_MT5_LOGIN')
        if PEPPERSTONE_MT5_PASSWORD == 'your_password_here':
            missing_vars.append('PEPPERSTONE_MT5_PASSWORD')
    
    # Check FXCM credentials if in BROKERS list
    if "FXCM" in BROKERS:
        if FXCM_API_TOKEN == 'your_fxcm_token_here':
            missing_vars.append('FXCM_API_TOKEN')
    
    if missing_vars:
        print(f"⚠️ Missing required environment variables: {missing_vars}")
        print("📝 Please check your .env file and ensure all credentials are set.")
        print("💡 Copy .env.example to .env and fill in your actual credentials.")
        return False
    
    print("✅ Configuration validation passed!")
    return True

# Run validation when module is imported
if __name__ == "__main__":
    validate_config()