# test_pepperstone_order.py
import MetaTrader5 as mt5
from src.utils.config import *
from src.trading.executor import PepperstoneExecutor

# Test the executor
try:
    executor = PepperstoneExecutor(
        PEPPERSTONE_MT5_TERMINAL,
        PEPPERSTONE_MT5_LOGIN,
        PEPPERSTONE_MT5_PASSWORD,
        PEPPERSTONE_MT5_SERVER
    )
    
    # Test EURUSD order
    print("\nTesting EURUSD order...")
    result = executor.place_order(
        signal=1,  # Buy
        symbol="EUR_USD",
        units=1000,  # 0.01 lots
        price=1.08500
    )
    
    if result and result.get('retcode') == 10009:
        print("✅ EURUSD order successful!")
    else:
        print(f"❌ EURUSD order failed: {result}")
    
    # Test XAUUSD order
    print("\nTesting XAUUSD order...")
    result = executor.place_order(
        signal=-1,  # Sell
        symbol="XAU_USD",
        units=1000,
        price=2650.00
    )
    
    if result and result.get('retcode') == 10009:
        print("✅ XAUUSD order successful!")
    else:
        print(f"❌ XAUUSD order failed: {result}")
        
except Exception as e:
    print(f"Error: {e}")