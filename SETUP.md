# Detailed Setup Guide for ICT-ML-Trading

This guide provides step-by-step instructions to get your ICT-ML-Trading system running from scratch.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [Broker Account Setup](#broker-account-setup)
4. [MT5 Terminal Setup](#mt5-terminal-setup)
5. [Configuration](#configuration)
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

## Configuration

### 1. Edit config.py

```bash
cd src/utils
notepad config.py
```

Update with your credentials:

```python
# OANDA Credentials
OANDA_API_TOKEN = "your-token-here"
OANDA_ACCOUNT_ID = "101-001-12345678-001"
OANDA_ENV = "practice"  # Keep as practice for demo

# FTMO MT5 Settings
FTMO_MT5_TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
FTMO_MT5_LOGIN = 12345678  # Your MT5 login number
FTMO_MT5_PASSWORD = "your-password"
FTMO_MT5_SERVER = "FTMO-Demo"

# Pepperstone MT5 Settings
PEPPERSTONE_MT5_TERMINAL = r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe"
PEPPERSTONE_MT5_LOGIN = 87654321
PEPPERSTONE_MT5_PASSWORD = "your-password"
PEPPERSTONE_MT5_SERVER = "Pepperstone-Demo"

# Active Brokers (comment out any you don't have)
BROKERS = ["OANDA", "FTMO", "PEPPERSTONE"]

# Instruments to trade
SYMBOLS = ["EUR_USD", "GBP_USD", "XAU_USD"]  # Start with these
```

### 2. Create Required Directories

```bash
mkdir data
mkdir checkpoints
mkdir reports
mkdir logs
```

## Data Preparation

### Option 1: Download from Dukascopy

1. Visit [Dukascopy Historical Data](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
2. Select:
   - Instrument: EUR/USD
   - Period: 1 Hour
   - Date range: Last 2 years
3. Download as CSV
4. Save as `data/EURUSD=X_60m.csv`

Repeat for other instruments:
- GBPUSD → `data/GBPUSD=X_60m.csv`
- XAUUSD → `data/XAUUSD=X_60m.csv`

### Option 2: Use yfinance (Automatic)

The system will automatically download data if CSV files are not found.

### Data Format Example

Your CSV should look like:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0950,1.0955,1.0945,1.0952,1250
2024-01-01 01:00:00,1.0952,1.0958,1.0950,1.0956,1180
```

## First Run

### 1. Test Data Loading

```python
# test_data.py
from src.data_processing.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data("EURUSD", "2024-01-01", "2024-12-31", "60m")
print(f"Loaded {len(df)} bars")
print(df.head())
```

### 2. Test Broker Connections

```python
# test_brokers.py
import MetaTrader5 as mt5
from src.utils.config import *

# Test OANDA
from src.data_processing.oanda_data import OandaDataFetcher
fetcher = OandaDataFetcher()
df = fetcher.fetch_ohlc("EUR_USD", "H1", 10)
print(f"OANDA: {len(df)} bars fetched")

# Test MT5
if mt5.initialize(FTMO_MT5_TERMINAL):
    print("FTMO MT5: Connected")
    mt5.shutdown()
```

### 3. Train Models (First Time Only)

```bash
python run_pipeline.py
```

This will take 5-6 hours. You'll see:
- Feature engineering progress
- Model training for each algorithm
- Cross-validation results
- Final model saved to `checkpoints/`

### 4. Start Live Trading

```bash
# Make sure MT5 terminals are open and logged in
python live_trade_multiinstrument.py
```

## Verification

### Check Logs

```bash
type logs\live_trade.log
```

You should see:
- "Configured brokers: ['OANDA', 'FTMO', 'PEPPERSTONE']"
- "Loading model: XXXUSD=X_60m_best_pipeline_xgb.pkl"
- "✅ OANDA executed EUR_USD trade"

### Check Trades

1. **OANDA**: Login to web platform, check positions
2. **MT5 Terminals**: Check "Trade" tab for open positions
3. **Logs**: All trades are logged with timestamps

### Performance Monitoring

```python
# check_performance.py
import pandas as pd
import matplotlib.pyplot as plt

# Read logs and plot daily P&L
# (implement based on your needs)
```

## Common Issues

### Issue: "IPC timeout" for MT5

**Solution**:
1. Make sure MT5 terminal is running
2. Check login credentials
3. Enable "Algo Trading" button
4. Run as Administrator

### Issue: "Symbol not found"

**Solution**:
1. Check exact symbol name in MT5
2. Update symbol mapping in executor.py:
```python
MT5_SYMBOL_MAP = {
    "EUR_USD": "EURUSD",  # Adjust to your broker's naming
    "US30_USD": "US30.cash",  # Some use .cash suffix
}
```

### Issue: "No data found"

**Solution**:
1. Check CSV file exists in data/ folder
2. Verify timestamp format
3. Ensure at least 120 bars of data

### Issue: Training interrupted

**Solution**:
```bash
# Just run again - it will resume from checkpoint
python run_pipeline.py
```

### Issue: Low accuracy on live trading

**Solution**:
1. Ensure you have recent data (within last month)
2. Retrain models quarterly
3. Start with small position sizes

## Next Steps

1. **Paper Trade First**: Run for at least 2 weeks on demo
2. **Monitor Daily**: Check logs and broker accounts
3. **Adjust Position Sizes**: Based on account size and risk
4. **Regular Retraining**: Every 3-6 months with new data
5. **Add More Instruments**: Gradually expand to more pairs

## Support

For issues:
1. Check error messages in `logs/live_trade.log`
2. Verify all credentials are correct
3. Ensure all prerequisites are installed
4. Open a GitHub issue with error details

---

Remember: **Always test on demo accounts first!**