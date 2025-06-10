# ICT-ML-Trading

An advanced machine learning-based forex trading system implementing Inner Circle Trader (ICT) concepts for multi-broker automated trading with 89-91% accuracy.

## 🚀 Overview

ICT-ML-Trading is a comprehensive trading framework that:
- **Implements ICT Concepts**: Market structure analysis, PD Arrays, Order Blocks, Fair Value Gaps, Liquidity concepts, and more
- **Uses Machine Learning**: XGBoost, Random Forest, Gradient Boosting, and Logistic Regression models with 89-91% accuracy
- **Multi-Broker Support**: Trades simultaneously on OANDA, FTMO, Pepperstone, and any MT5-compatible broker
- **Real-Time Trading**: Executes trades every hour based on ML predictions with 5-bar window logic
- **Risk Management**: Standardized position sizing (10,000 units = 0.1 lot), optional TP/SL settings
- **Backtesting**: Comprehensive backtesting with performance metrics
- **Secure Configuration**: Environment variable-based credential management

## 📊 Performance

Based on extensive testing:
- **Model Accuracy**: 89-91% (XGBoost performs best)
- **Win Rate**: ~65-70%
- **Sharpe Ratio**: 2.0+ (annualized)
- **Maximum Drawdown**: <15%

## 🏗️ Architecture

### Core Components

1. **Data Processing**
   - Multi-timeframe data alignment (1H, 1D, 1W)
   - Support for Dukascopy CSV format
   - Real-time OANDA API integration
   - Automatic data validation and cleaning
   - Optional data collection script for automated downloads

2. **Feature Engineering (ICT Concepts)**
   - **Market Structure**: HH/HL/LH/LL detection, trend classification, MSS/BOS
   - **PD Arrays**: Order blocks, Fair Value Gaps, Breaker blocks, Mitigation blocks
   - **Liquidity**: BSL/SSL levels, liquidity pools, stop runs
   - **Time Features**: Trading sessions (London, NY, Asia), kill zones
   - **Patterns**: OTE, Turtle Soup, Breaker patterns
   - **Technical Indicators**: ATR, RSI, Bollinger Bands, Moving Averages

3. **Machine Learning Pipeline**
   - Symbol-specific model training and loading
   - Automated feature selection
   - Walk-forward analysis with embargo
   - Nested cross-validation
   - Model checkpointing for interrupted training
   - Support for multiple models (XGBoost, RF, GB, LR, SVM)

4. **Trading Execution**
   - Signal generation from symbol-specific ML predictions
   - Multi-broker execution (simultaneous)
   - 5-bar window position management
   - State persistence across restarts
   - Automatic position closing after 5 hours
   - Standardized position sizing across all brokers

## 📋 Prerequisites

- **Python 3.10+**
- **Windows 11** (recommended for MT5 support)
- **MetaTrader 5** (for FTMO/Pepperstone)
- **Broker Demo/Live Accounts**
- **8GB+ RAM** (for model training)
- **10GB+ disk space** (for data and models)

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ICT-ML-Trading.git
cd ICT-ML-Trading
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install MetaTrader5 (for MT5 brokers)

```bash
pip install MetaTrader5
```

Note: MT5 package only works on Windows. For Linux/Mac, use only API-based brokers like OANDA.

## ⚙️ Configuration

### 1. Environment Setup (Secure Credentials)

The system now uses environment variables for secure credential management:

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your actual broker credentials
```

**Edit `.env` file with your real credentials:**

```bash
# === OANDA API Credentials ===
OANDA_API_TOKEN=your_real_oanda_api_token_here
OANDA_ACCOUNT_ID=your_real_oanda_account_id_here
OANDA_ENV=practice

# === FTMO MT5 Credentials ===
FTMO_MT5_LOGIN=your_real_ftmo_login_number
FTMO_MT5_PASSWORD=your_real_ftmo_password
FTMO_MT5_SERVER=FTMO-Demo

# === Pepperstone MT5 Credentials ===
PEPPERSTONE_MT5_LOGIN=your_real_pepperstone_login_number
PEPPERSTONE_MT5_PASSWORD=your_real_pepperstone_password
PEPPERSTONE_MT5_SERVER=Pepperstone-Demo

# === Risk Settings ===
USE_TP_SL=False
TAKE_PROFIT_PIPS=10
STOP_LOSS_PIPS=5
```

**Important**: Never commit the `.env` file to version control. It contains your sensitive credentials and is automatically ignored by git.

### 2. Broker Configuration

The active brokers and instruments are configured in `src/utils/config.py`:

```python
# Active brokers (only configured brokers will be used)
BROKERS = ["OANDA", "FTMO", "PEPPERSTONE"]

# Instruments to trade
SYMBOLS = ["EUR_USD", "GBP_USD", "XAU_USD", "US30_USD", "NAS100_USD"] #add as many as you will like. but you will need to train models for each
```

The system automatically detects which brokers have valid credentials and only trades with properly configured brokers.

### 3. Validate Configuration

Test your setup:

```bash
python -c "from src.utils.config import validate_config; validate_config()"
```

This will verify that all required credentials are properly set for your active brokers.

### 4. MT5 Terminal Setup

For MT5 brokers (FTMO, Pepperstone):
1. Install MT5 terminal from broker
2. Enable automated trading: Tools → Options → Expert Advisors → Allow automated trading
3. Login to your demo/live account
4. Keep terminal open while running the bot

## 📊 Data Preparation

### Option 1: Use Included Data Collection Script (Recommended)

For automated data collection, you can use the included data collection script:

```bash
python collect_data.py  # If you have this script available
```

This script will automatically download and format the required historical data for all configured symbols.

### Option 2: Manual Data Setup

#### Historical Data Format

Place CSV files in `data/` folder with format:
```
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0950,1.0955,1.0945,1.0952,1250
```

#### Download from Dukascopy

1. Visit [Dukascopy Historical Data](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
2. Select instrument and date range
3. Download as CSV
4. Place in `data/` folder as `SYMBOL_60m.csv` (e.g., `EURUSD=X_60m.csv`)

**Required file naming convention:**
- `EURUSD=X_60m.csv` for EUR_USD
- `GBPUSD=X_60m.csv` for GBP_USD
- `XAUUSD=X_60m.csv` for XAU_USD
- `USA30=X_60m.csv` for US30_USD
- `USATECH=X_60m.csv` for NAS100_USD

## 🚂 Training Models

### 1. First-Time Training (5-6 hours)

```bash
python train_models.py
```

This will:
- Load and align multi-timeframe data for all symbols
- Engineer 50+ ICT features for each symbol
- Train multiple models with nested CV for each symbol
- Save the best model for each symbol to `checkpoints/`
- Generate performance reports in `reports/`

**Key Features:**
- **Symbol-Specific Models**: Each trading pair gets its own optimized model
- **Checkpoint Recovery**: Resume training if interrupted
- **Comprehensive Evaluation**: Walk-forward testing with time-series validation
- **Model Selection**: Automatic best-performing model selection per symbol

### 2. Using Pre-trained Models

If training was completed before, symbol-specific models are loaded from checkpoints automatically. The system includes a ModelManager that:
- Maps trading symbols to their respective model files
- Loads models on-demand for efficiency
- Validates model availability before trading

## 🤖 Live Trading

### 1. Start Trading Bot

```bash
python live_trading_bot.py
```

**Enhanced Features** in the latest version:
- **5-Bar Window Logic**: Trades are held for exactly 5 hours (matching ML model's prediction window)
- **State Persistence**: Survives restarts without duplicate trades using `trade_state.json`
- **Smart Entry Validation**: Prevents chasing extended moves after restarts
- **Automatic Position Management**: Closes trades after 5 hours automatically
- **Symbol-Specific Models**: Uses the best model for each trading pair
- **Standardized Position Sizing**: 10,000 units = 0.1 lot consistently across all brokers

### 2. How It Works

The bot will:
- Load symbol-specific models from checkpoints for all configured instruments
- Connect to all configured brokers (skips unconfigured ones gracefully)
- Fetch latest market data every hour at :01
- Generate signals based on 5-bar forward prediction using the appropriate model
- Execute trades with proper window management
- Maintain state in `trade_state.json` for persistence
- Automatically close positions after 5 hours
- Log all activities to `logs/live_trade.log`

### 3. Position Management Logic

**5-Bar Window System:**
- **Entry**: New positions opened based on ML signals
- **Hold Period**: Positions held for exactly 5 hours (5 bars on 1H timeframe)
- **Auto-Close**: Positions automatically closed after 5 hours
- **Same Direction**: No new trades in same direction within 5-hour window
- **Opposite Direction**: Allowed (closes existing position first)

**State Persistence:**
- Active trades stored in `trade_state.json`
- System remembers positions across restarts
- Prevents duplicate trades after interruptions
- Validates price movement for late entries

### 4. Monitor Performance

- Check broker terminals for executed trades
- Review `logs/live_trade.log` for detailed execution info
- Monitor `trade_state.json` for active positions
- Verify 5-hour automatic closures
- Track individual symbol performance

### 5. Stop Trading

Press `Ctrl+C` to stop the bot gracefully. State is preserved in `trade_state.json` for seamless restart.

## 📈 Backtesting

Backtesting is integrated into the training pipeline (`train_models.py`). Results include:
- Equity curve visualization per symbol
- Maximum drawdown analysis
- Trade-by-trade performance
- Sharpe ratio and other metrics
- Symbol-specific performance attribution

## 🔧 Advanced Analysis

For deeper performance analysis:

```bash
python advanced_analysis.py
```

This provides:
- Monte Carlo bootstrap analysis
- Regime-based walk-forward testing
- Advanced performance attribution
- Cross-symbol correlation analysis

## 🛠️ Troubleshooting

### Environment Variable Issues

**Error**: "Missing required environment variables"
- Ensure `.env` file exists and contains all required credentials
- Check that variable names match exactly (case-sensitive)
- Verify no extra spaces around the `=` sign in `.env`

### MT5 Connection Issues

**Error**: "IPC timeout"
- Ensure MT5 terminal is running and logged in
- Enable "Algo Trading" button (should be green)
- Check Tools → Options → Expert Advisors settings
- Run as Administrator if needed

### Symbol Not Found

**Error**: "Symbol XYZ not found"
- Check exact symbol name in MT5 Market Watch
- Update symbol mapping in executor classes
- Some brokers use different naming (e.g., "US30" vs "US30.cash")

### Model Loading Issues

**Error**: "No model file found for symbol"
- Ensure training completed successfully for all symbols
- Check `checkpoints/` directory for model files
- Verify file naming matches symbol mapping in ModelManager
- Re-run `train_models.py` if models are missing

### Multiple MT5 Instances

The system handles multiple MT5 connections automatically by:
- Reinitializing connection for each broker
- Proper connection cleanup between trades
- Automatic retry on connection failure

### Missing Data

**Error**: "Insufficient data"
- Ensure CSV files have at least 120 bars
- Check data format matches expected columns
- Verify timestamp format is correct
- Use the data collection script for automated data preparation

## 📁 Project Structure

```
ICT-ML-Trading/
├── data/                    # Historical price data
├── checkpoints/             # Saved symbol-specific models
├── reports/                # Performance reports
├── logs/                   # Trading logs
├── src/
│   ├── data_processing/    # Data loading and validation
│   ├── features/           # ICT feature engineering
│   ├── ml_models/          # Model training and evaluation
│   ├── trading/            # Strategy and execution
│   └── utils/              # Configuration and helpers
├── train_models.py         # Model training pipeline
├── live_trading_bot.py     # Live trading execution
├── advanced_analysis.py    # Extended analysis tools
├── .env.example           # Environment template
├── .env                   # Your credentials (not committed)
└── requirements.txt       # Dependencies
```

## 🎯 ICT Concepts Implemented

- **Market Structure**: Identifies swing highs/lows, trends, and structure shifts
- **Order Blocks**: Last opposing candle before aggressive move
- **Fair Value Gaps**: Price inefficiencies in 3-candle patterns
- **Breaker Blocks**: Failed order blocks that reverse
- **Liquidity Concepts**: Buy/Sell-side liquidity, stop runs
- **Optimal Trade Entry (OTE)**: 62-79% Fibonacci retracement zones
- **Power of 3**: Accumulation, Manipulation, Distribution phases
- **Kill Zones**: High-probability time windows for trades

## ⚡ Performance Tips

1. **VPS Deployment**: Run on a VPS near broker servers for lower latency
2. **Resource Allocation**: Close unnecessary programs during training
3. **Data Quality**: Use high-quality tick data for better features
4. **Model Selection**: XGBoost typically performs best
5. **Position Sizing**: Start small and scale up gradually (system uses standardized 10k units)
6. **Symbol Selection**: Focus on high-accuracy symbols from training reports

## 🔒 Security Best Practices

1. **Environment Variables**: All credentials stored in `.env` file (never committed)
2. **Git Security**: `.env` automatically ignored by `.gitignore`
3. **2FA**: Enable 2FA on all broker accounts
4. **Monitor Activity**: Regularly check account activity
5. **Position Limits**: Set maximum position limits with brokers
6. **Demo Testing**: Always test on demo accounts first

## ⚠️ Risk Disclaimer

**IMPORTANT**: Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The high degree of leverage can work against you as well as for you. Before deciding to trade foreign exchange, carefully consider your investment objectives, experience level, and risk tolerance.

This software is provided "as is" without warranty of any kind. The authors are not responsible for any financial losses incurred through the use of this software. Always test thoroughly on demo accounts before risking real capital.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Currently not accepting contributions. For issues or questions, please open a GitHub issue.

## 📞 Support

- GitHub Issues: [Create an issue](https://github.com/OPKYEI/ICT-ML-Trading/issues)
- Documentation: Check the additional documentation files for detailed guides

---

**Happy Trading! 📈** Remember to always trade responsibly and within your risk tolerance.
