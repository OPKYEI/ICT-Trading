# test_mt5_simple.py
import os
import MetaTrader5 as mt5

# Check if terminals exist
ftmo_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
pepper_path = r"C:\Pepperstone MetaTrader 5\terminal64.exe"

print(f"FTMO terminal exists: {os.path.exists(ftmo_path)}")
print(f"Pepperstone terminal exists: {os.path.exists(pepper_path)}")

# Try basic initialization without credentials
print("\nTrying basic MT5 init...")
if mt5.initialize():
    print("✅ MT5 initialized with default terminal")
    info = mt5.terminal_info()
    print(f"Terminal: {info.path}")
    print(f"Data path: {info.data_path}")
    mt5.shutdown()
else:
    print(f"❌ Failed: {mt5.last_error()}")