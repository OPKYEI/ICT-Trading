# Detailed Setup Guide for ICT-ML-Trading

This guide provides step-by-step instructions to get your ICT-ML-Trading system running from scratch with the latest security and automation features.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [Broker Account Setup](#broker-account-setup)
4. [MT5 Terminal Setup](#mt5-terminal-setup)
5. [Secure Configuration Setup](#secure-configuration-setup)
6. [Data Preparation](#data-preparation)
7. [First Run](#first-run)
8. [Verification](#verification)
9. [Common Issues](#common-issues)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (required for MT5 support)
- **CPU**: 4 cores (8+ recommended for training)
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 20GB free space
- **Internet**: Stable broadband connection
- **Python**: 3.10 or higher

### Recommended Setup
- **OS**: Windows 11 Pro
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7)
- **RAM**: 16-32GB
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA GPU (optional, for faster XGBoost)

## Python Environment Setup

### 1. Install Python

Download Python 3.10+ from [python.org](https://www.python.org/downloads/)

During installation:
- ✅ Check "Add Python to PATH"
- ✅ Check "Install pip"
- Choose "Customize installation"
- ✅ Check all optional features

### 2. Verify Installation

```bash
python --version  # Should show Python 3.10.x or higher
pip --version     # Should show pip 22.x or higher
```

### 3. Create Project Directory

```bash
mkdir C:\ICT-ML-Trading
cd C:\ICT-ML-Trading
git clone https://github.com/yourusername/ICT-ML-Trading.git .
```

### 4. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate

# You should see (venv) in your terminal
```

### 5. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If MetaTrader5 fails to install:
```bash
# Install Visual C++ Redistributable first
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
pip install MetaTrader5
```

### 6. Install python-dotenv (for environment variables)

```bash
pip install python-dotenv
```

## Broker Account Setup

### OANDA Setup

1. **Create Demo Account**
   - Visit [OANDA Demo](https://www.oanda.com/forex-trading/demo-forex-account/)
   - Register for a practice account
   - Note your account ID

2. **Get API Token**
   - Login to [OANDA Hub](https://hub.oanda.com/)
   - Go to "Manage API Access"
   - Generate a practice API token
   - Copy the token (looks like: `xxxxx-xxxxx-xxxxx`)

3. **Find Account ID**
   - In OANDA Hub, go to "My Services"
   - Copy your account number (format: `xxx-xxx-xxxxxxx-xxx`)

### FTMO Setup

1. **Create FTMO Account**
   - Visit [FTMO](https://ftmo.com/)
   - Register and get a demo account
   - Download FTMO MT5 Terminal

2. **Install FTMO MT5**
   - Run the installer
   - Default path: `C:\Program Files\FTMO Global Markets MT5 Terminal`
   - Login with demo credentials

### Pepperstone Setup

1. **Create Pepperstone Demo**
   - Visit [Pepperstone](https://pepperstone.com/)
   - Open a demo account
   - Download Pepperstone MT5

2. **Install Pepperstone MT5**
   - Run the installer
   - Default path: `C:\Program Files\Pepperstone MetaTrader 5`
   - Login with demo credentials

## MT5 Terminal Setup

### For Each MT5 Terminal:

1. **Enable Automated Trading**
   ```
   Tools → Options → Expert Advisors
   ✅ Allow automated trading
   ✅ Allow DLL imports
   ✅ Disable news
   ```

2. **Configure Charts**
   ```
   File → Open Chart
   Add: EURUSD, GBPUSD, XAUUSD, US30, NAS100
   Set all to H1 (1 hour) timeframe
   ```

3. **Verify Symbols**
   ```
   View → Market Watch (Ctrl+M)
   Right-click → Show All
   Note exact symbol names
   ```

4. **Keep Terminal Running**
   - Minimize (don't close) during trading
   - Disable Windows sleep mode

## Secure Configuration Setup

### 1. Create Environment File (Secure Credentials)

**IMPORTANT**: The system now uses environment variables for security. Your credentials are never stored in code.

```bash
# Copy the secure template
cp .env.example .env
```

### 2. Edit .env with Your Real Credentials

```bash
# Open .env file in notepad
notepad .env
```

**Fill in your actual credentials** (replace the placeholder values):

```bash
# === Risk & TP/SL Settings ===
USE_TP_SL=False
TAKE_PROFIT_PIPS=10
STOP_LOSS_PIPS=5

# === OANDA API Credentials ===
OANDA_API_TOKEN=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6  # Your real OANDA token
OANDA_ACCOUNT_ID=101-001-12345678-001               # Your real account ID
OANDA_ENV=practice

# === FXCM Credentials (if you have them) ===
FXCM_API_TOKEN=your_real_fxcm_token_here

# === FTMO MT5 Credentials ===
FTMO_MT5_TERMINAL=C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe
FTMO_MT5_LOGIN=12345678                    # Your real FTMO login number
FTMO_MT5_PASSWORD=YourRealPassword123      # Your real FTMO password
FTMO_MT5_SERVER=FTMO-Demo

# === Pepperstone MT5 Credentials ===
PEPPERSTONE_MT5_TERMINAL=C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe
PEPPERSTONE_MT5_LOGIN=87654321             # Your real Pepperstone login
PEPPERSTONE_MT5_PASSWORD=YourPepperstonePass123  # Your real Pepperstone password
PEPPERSTONE_MT5_SERVER=Pepperstone-Demo
```

**Security Notes:**
- ✅ The `.env` file is automatically ignored by git (never committed)
- ✅ Your real credentials never appear in the source code
- ✅ Safe to share your repository without exposing secrets

### 3. Configure Active Brokers

Edit `src/utils/config.py` to specify which brokers to use:

```python
# Active Brokers (only add brokers you have credentials for)
BROKERS = ["OANDA", "FTMO", "PEPPERSTONE"]  # Remove any you don't have

# Instruments to trade (start with fewer for testing)
SYMBOLS = ["EUR_USD", "GBP_USD", "XAU_USD"]  # Add more later
```

### 4. Validate Configuration

**Test your setup**:

```bash
python -c "from src.utils.config import validate_config; validate_config()"
```

**Expected output for properly configured system:**
```
✅ Loaded environment variables from: C:\...\ICT-ML-Trading\.env
✅ Configuration validation passed!
```

**If you see warnings:**
```
⚠️ Missing required environment variables: ['FXCM_API_TOKEN']
```
This is normal if you don't have FXCM credentials. Just remove "FXCM" from the BROKERS list in config.py.

### 5. Create Required Directories

```bash
mkdir data
mkdir checkpoints
mkdir reports
mkdir logs
```

## Data Preparation

### Option 1: Use Data Collection Script (Recommended)

If you have the optional data collection script:

```bash
python collect_data.py
```

This will automatically download and format all required data files.

### Option 2: Download from Dukascopy (Manual)

1. Visit [Dukascopy Historical Data](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
2. Select:
   - Instrument: EUR/USD
   - Period: 1 Hour
   - Date range: Last 2 years
3. Download as CSV
4. Save as `data/EURUSD=X_60m.csv`

**Required file naming** (must match exactly):
- EUR_USD → `data/EURUSD=X_60m.csv`
- GBP_USD → `data/GBPUSD=X_60m.csv`
- XAU_USD → `data/XAUUSD=X_60m.csv`
- US30_USD → `data/USA30=X_60m.csv`
- NAS100_USD → `data/USATECH=X_60m.csv`

### Option 3: Automatic Download (Fallback)

The system will automatically download data using yfinance if CSV files are not found.

### Data Format Example

Your CSV should look like:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0950,1.0955,1.0945,1.0952,1250
2024-01-01 01:00:00,1.0952,1.0958,1.0950,1.0956,1180
```

## First Run

### 1. Test Configuration Loading

```python
# test_config.py
from src.utils.config import validate_config, OANDA_API_TOKEN, SYMBOLS, BROKERS

validate_config()
print(f"Active brokers: {BROKERS}")
print(f"Trading symbols: {SYMBOLS}")
print(f"OANDA token loaded: {OANDA_API_TOKEN[:10]}...")
```

### 2. Test Data Loading

```python
# test_data.py
from src.data_processing.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data("EURUSD", "2024-01-01", "2024-12-31", "60m")
print(f"Loaded {len(df)} bars")
print(df.head())
```

### 3. Test Broker Connections

```python
# test_brokers.py
import MetaTrader5 as mt5
from src.utils.config import *

# Test OANDA
try:
    from src.data_processing.oanda_data import OandaDataFetcher
    fetcher = OandaDataFetcher()
    df = fetcher.fetch_ohlc("EUR_USD", "H1", 10)
    print(f"✅ OANDA: {len(df)} bars fetched")
except Exception as e:
    print(f"❌ OANDA failed: {e}")

# Test FTMO MT5
if FTMO_MT5_LOGIN != 0:
    if mt5.initialize(FTMO_MT5_TERMINAL):
        login_result = mt5.login(FTMO_MT5_LOGIN, FTMO_MT5_PASSWORD, FTMO_MT5_SERVER)
        if login_result:
            print("✅ FTMO MT5: Connected successfully")
        else:
            print(f"❌ FTMO MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
    else:
        print("❌ FTMO MT5: Failed to initialize terminal")
```

### 4. Train Models (First Time Only - 5-6 hours)

```bash
python train_models.py
```

**What happens during training:**
- ✅ Loads data for all symbols in SYMBOLS list
- ✅ Engineers 50+ ICT features for each symbol
- ✅ Trains multiple ML models (XGBoost, Random Forest, etc.)
- ✅ Selects best model for each symbol
- ✅ Saves symbol-specific models to `checkpoints/`
- ✅ Generates performance reports in `reports/`

**Progress indicators:**
```
🔄 Processing EURUSD...
✅ EURUSD feature engineering completed: 52 features
🤖 Training XGBoost for EURUSD...
📊 EURUSD - XGBoost accuracy: 91.2%
💾 EURUSD best model saved: EURUSD=X_60m_best_pipeline_xgb.pkl
```

### 5. Start Live Trading

```bash
# Make sure all MT5 terminals are open and logged in
python live_trading_bot.py
```

**What happens on startup:**
```
🚀 Starting improved live_trade system with symbol-specific models
📋 Features:
   - Symbol-specific model loading
   - Trades valid for exactly 5 bars
   - Automatic position closing after 5 hours
   - State persistence across restarts
   - Standardized position sizing (10,000 units = 0.1 lot)

✅ Loaded environment variables from: C:\...\ICT-ML-Trading\.env
📊 Active brokers: ['OANDA', 'FTMO', 'PEPPERSTONE']
🤖 Preloading models for all symbols...
✅ Loaded model for EUR_USD: EURUSD=X_60m_best_pipeline_xgb.pkl
✅ Loaded model for GBP_USD: GBPUSD=X_60m_best_pipeline_xgb.pkl
✅ Model loading completed
▶️ Running trading cycle
```

## Verification

### Check System Status

```bash
# Validate configuration
python -c "from src.utils.config import validate_config; validate_config()"

# Check model availability
python -c "
from live_trading_bot import ModelManager
mm = ModelManager()
mm.preload_all_models(['EUR_USD', 'GBP_USD', 'XAU_USD'])
"
```

### Check Logs

```bash
type logs\live_trade.log
```

**Successful startup logs:**
```
2024-01-15 09:01:05 - live_trade - INFO - ✅ Initialized OANDA executor
2024-01-15 09:01:06 - live_trade - INFO - ✅ Initialized FTMO executor
2024-01-15 09:01:07 - live_trade - INFO - 📊 Active brokers: ['OANDA', 'FTMO']
2024-01-15 09:01:08 - live_trade - INFO - EUR_USD: signal 1 @ 1.0952 (model: EURUSD=X_60m)
2024-01-15 09:01:09 - live_trade - INFO - ✅ OANDA executed EUR_USD trade: order 12345
```

### Check Trade State

```bash
type trade_state.json
```

**Active trades format:**
```json
{
  "active_trades": {
    "EUR_USD": {
      "signal": 1,
      "entry_time": "2024-01-15 09:01:00",
      "entry_price": 1.0952,
      "position_tickets": {
        "OANDA": null,
        "FTMO": 98765432
      }
    }
  },
  "last_check": "2024-01-15 09:01:00"
}
```

### Check Trades in Brokers

1. **OANDA**: Login to web platform → Positions tab
2. **FTMO MT5**: Trade tab → check for open positions
3. **Pepperstone MT5**: Trade tab → verify position sizes

### Verify Position Sizing

**All brokers should show consistent position sizes:**
- 10,000 units = 0.1 lot
- 100,000 units = 1.0 lot

This ensures standardized risk across all brokers.

## Common Issues

### Issue: "Missing required environment variables"

**Solution**:
1. Ensure `.env` file exists in project root
2. Check all required variables are set
3. No spaces around `=` in `.env`
4. Remove unused brokers from BROKERS list

```bash
# Check if .env exists
dir .env

# Verify content
type .env
```

### Issue: "No model file found for symbol"

**Solution**:
1. Run training first: `python train_models.py`
2. Check `checkpoints/` directory for model files
3. Verify symbol naming matches ModelManager mapping

```bash
# Check available models
dir checkpoints\*.pkl

# Expected files:
# EURUSD=X_60m_best_pipeline_xgb.pkl
# GBPUSD=X_60m_best_pipeline_xgb.pkl
```

### Issue: "IPC timeout" for MT5

**Solution**:
1. Make sure MT5 terminal is running and logged in
2. Check login credentials in `.env`
3. Enable "Algo Trading" button (should be green)
4. Run terminal as Administrator
5. Verify correct terminal path in `.env`

### Issue: "Symbol not found" in MT5

**Solution**:
1. Check exact symbol name in MT5 Market Watch
2. Update symbol mapping in ModelManager if needed
3. Some brokers use different naming:

```python
# In live_trading_bot.py, update symbol mapping if needed
symbol_mapping = {
    "US30_USD": "US30.cash",    # Some brokers use .cash
    "NAS100_USD": "NAS100.m",   # Some use .m suffix
}
```

### Issue: "No data found"

**Solution**:
1. Check CSV file exists: `dir data\*.csv`
2. Verify file naming matches required format
3. Use data collection script if available
4. Ensure at least 120 bars of data

### Issue: Training interrupted

**Solution**:
```bash
# Resume training from checkpoints
python train_models.py

# Check logs for progress
type logs\training.log
```

### Issue: Live trading not executing trades

**Solution**:
1. Check signal generation: review logs for "signal X @ Y"
2. Verify broker connections: look for "✅ Initialized X executor"
3. Check 5-bar window logic: existing trades prevent new ones
4. Verify sufficient account balance

```python
# Test signal generation manually
from live_trading_bot import run_once
run_once()  # Run one trading cycle
```

### Issue: Different position sizes across brokers

**Solution**:
The system uses standardized sizing (10,000 units). If you see differences:
1. Check broker specifications in MT5
2. Some indices may use different contract sizes
3. Verify executor implementations handle sizing correctly

## Performance Monitoring

### Daily Checks

```bash
# Check recent logs
type logs\live_trade.log | findstr /i "executed"

# Monitor active trades
type trade_state.json

# Check for errors
type logs\live_trade.log | findstr /i "error"
```

### Weekly Analysis

```bash
# Run extended analysis
python advanced_analysis.py

# Check model performance
python -c "
import pandas as pd
import glob
reports = glob.glob('reports/*.html')
print(f'Available reports: {reports}')
"
```

## Next Steps

1. **Paper Trade First**: Run for at least 2 weeks on demo accounts
2. **Monitor Daily**: Check logs and broker positions daily
3. **Validate Performance**: Compare expected vs actual results
4. **Scale Gradually**: Increase position sizes after successful testing
5. **Regular Retraining**: Retrain models every 3-6 months
6. **Add Instruments**: Gradually expand to more pairs after validation

## Security Checklist

- ✅ `.env` file created with real credentials
- ✅ `.env` not committed to git (verify with `git status`)
- ✅ Only demo accounts used for initial testing
- ✅ 2FA enabled on all broker accounts
- ✅ Strong passwords for all accounts
- ✅ Regular monitoring of account activity

## Support

For issues:
1. Check error messages in `logs/live_trade.log`
2. Verify configuration with `validate_config()`
3. Ensure all prerequisites are installed
4. Run individual test scripts to isolate problems
5. Open a GitHub issue with:
   - Full error message
   - Steps to reproduce
   - System configuration
   - Log excerpts

---

**Remember: Always test extensively on demo accounts before risking real capital!**

The system includes multiple safety features:
- 5-hour position windows
- Standardized position sizing
- State persistence
- Comprehensive logging
- Graceful error handling

Start conservative and scale up as you gain confidence in the system's performance.