# find_mt5.py
import os
import glob

# Common MT5 installation locations
search_paths = [
    r"C:\Program Files\*MetaTrader*\terminal64.exe",
    r"C:\Program Files (x86)\*MetaTrader*\terminal64.exe",
    r"C:\Program Files\*MT5*\terminal64.exe",
    r"C:\Program Files (x86)\*MT5*\terminal64.exe",
    r"C:\Users\*\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe",
    r"C:\*\*MetaTrader*\terminal64.exe",
]

print("Searching for MT5 installations...\n")
for pattern in search_paths:
    for path in glob.glob(pattern, recursive=True):
        print(f"Found: {path}")