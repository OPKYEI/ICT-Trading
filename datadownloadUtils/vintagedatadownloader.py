#!/usr/bin/env python3
"""
BLAZINGLY FAST Forex Data Downloader
Uses Alpha Vantage API - downloads 20 years of data in seconds per symbol
"""

import requests
import os
import time
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

# Configuration
API_KEY = "YOUR_FREE_API_KEY_HERE"  # Get from: https://www.alphavantage.co/support/#api-key
OUTPUT_DIR = "./forex_data_ultrafast"
MAX_THREADS = 5  # Parallel downloads

# Global progress tracking
progress_lock = threading.Lock()
completed_symbols = 0
total_symbols = 0

def get_api_key():
    """Get API key from user if not set"""
    global API_KEY
    if API_KEY == "6PIKM8EMOB7TPLAW":
        print("🔑 You need a FREE API key from Alpha Vantage")
        print("   1. Go to: https://www.alphavantage.co/support/#api-key")
        print("   2. Get your free key (takes 30 seconds)")
        print("   3. Enter it below:")
        API_KEY = input("\nEnter your API key: ").strip()
    return API_KEY

def download_symbol_data(symbol, timeframe="60min"):
    """Download all historical data for one symbol - BLAZINGLY FAST"""
    global completed_symbols
    
    # Alpha Vantage forex URL - gets ALL historical data in one call
    if timeframe == "daily":
        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={symbol[:3]}&to_symbol={symbol[3:]}&outputsize=full&apikey={API_KEY}&datatype=csv"
    else:
        url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={symbol[:3]}&to_symbol={symbol[3:]}&interval={timeframe}&outputsize=full&apikey={API_KEY}&datatype=csv"
    
    start_time = time.time()
    print(f"📥 Downloading {symbol} ({timeframe})...")
    
    try:
        # Single HTTP request gets ALL historical data
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Save directly as CSV - no processing needed!
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Count rows to show progress
            row_count = response.text.count('\n') - 1  # Subtract header
            elapsed = time.time() - start_time
            
            with progress_lock:
                completed_symbols += 1
                print(f"✅ {symbol}: {row_count:,} rows in {elapsed:.1f}s ({completed_symbols}/{total_symbols})")
            
            return True
            
        else:
            print(f"❌ {symbol}: API error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {symbol}: {str(e)}")
        return False

def download_multiple_symbols(symbols, timeframe="60min"):
    """Download multiple symbols in parallel - MAXIMUM SPEED"""
    global total_symbols, completed_symbols
    
    total_symbols = len(symbols)
    completed_symbols = 0
    
    print(f"🚀 BLAZINGLY FAST FOREX DOWNLOADER")
    print(f"📈 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Timeframe: {timeframe}")
    print(f"🔗 Threads: {MAX_THREADS}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 50)
    
    start_time = time.time()
    
    # Download all symbols in parallel
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(download_symbol_data, symbol, timeframe) for symbol in symbols]
        
        # Wait for all downloads to complete
        results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    successful = sum(results)
    
    print("=" * 50)
    print(f"🎉 DOWNLOAD COMPLETE!")
    print(f"✅ Successful: {successful}/{total_symbols}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"⚡ Average: {total_time/len(symbols):.1f} seconds per symbol")
    print(f"📁 Files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Get API key
    api_key = get_api_key()
    
    # Popular forex pairs - edit this list
    SYMBOLS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "EURCHF", "GBPJPY", "GBPCHF", "AUDJPY", "AUDCHF",
        "NZDJPY", "EURAUD", "EURNZD", "AUDCAD", "AUDNZD", "CADJPY"
    ]
    
    # Choose timeframe
    print("\nSelect timeframe:")
    print("1. Hourly (60min) - Last 30 days")
    print("2. Daily - Up to 20 years")
    
    choice = input("Enter choice (1 or 2): ").strip()
    timeframe = "daily" if choice == "2" else "60min"
    
    # Confirm download
    print(f"\nReady to download {len(SYMBOLS)} symbols ({timeframe})")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm == 'y':
        download_multiple_symbols(SYMBOLS, timeframe)
    else:
        print("Download cancelled")