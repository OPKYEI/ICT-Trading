import MetaTrader5 as mt5
from src.utils.config import (
    PEPPERSTONE_MT5_TERMINAL, PEPPERSTONE_MT5_LOGIN,
    PEPPERSTONE_MT5_PASSWORD, PEPPERSTONE_MT5_SERVER
)

mt5.initialize(
    path=PEPPERSTONE_MT5_TERMINAL,
    login=PEPPERSTONE_MT5_LOGIN,
    password=PEPPERSTONE_MT5_PASSWORD,
    server=PEPPERSTONE_MT5_SERVER,
    portable=True,
    timeout=120,
)

# 1️⃣  Dump every symbol that contains “US” or “NAS”
for sym in mt5.symbols_get("*US*"):
    print(sym.name)

print("-----")
for sym in mt5.symbols_get("*NAS*"):
    print(sym.name)

print("last_error:", mt5.last_error())
mt5.shutdown()
