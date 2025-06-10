#!/usr/bin/env python3
"""
Complete standalone ultra-fast Dukascopy downloader
Usage: python ultra_fast_duka.py
"""

import asyncio
import aiohttp
import time
import csv
import os
import struct
from datetime import date, timedelta, datetime
from lzma import LZMADecompressor, LZMAError, FORMAT_AUTO
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict

# Configuration
MAX_CONCURRENT_REQUESTS = 30  # Aggressive but not crazy
TIMEOUT = 10  # Reasonable timeout
OUTPUT_DIR = "./data_ultrafast"

class TickData:
    def __init__(self, timestamp, ask, bid, ask_volume, bid_volume):
        self.timestamp = timestamp
        self.ask = ask
        self.bid = bid
        self.ask_volume = ask_volume
        self.bid_volume = bid_volume

class HourlyCandle:
    def __init__(self, timestamp, open_price, high, low, close_price, volume):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close_price
        self.volume = volume

async def download_single_file(session, url):
    """Download a single bi5 file with error handling"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as response:
            if response.status == 200:
                data = await response.read()
                return url, data
            return url, b''
    except Exception as e:
        return url, b''

def decompress_lzma(data):
    """Decompress LZMA data"""
    if not data or len(data) == 0:
        return b""
    
    results = []
    while True:
        decomp = LZMADecompressor(FORMAT_AUTO, None, None)
        try:
            res = decomp.decompress(data)
        except LZMAError:
            if results:
                break
            else:
                raise
        results.append(res)
        data = decomp.unused_data
        if not data:
            break
        if not decomp.eof:
            raise LZMAError("Compressed data ended before the end-of-stream marker was reached")
    return b"".join(results)

def parse_bi5_data(data, symbol, day, hour):
    """Parse bi5 binary data into tick objects"""
    if not data or len(data) == 0:
        return []
    
    try:
        # Decompress
        decompressed = decompress_lzma(data)
        if not decompressed:
            return []
        
        # Parse 20-byte records
        tick_size = 20
        num_ticks = len(decompressed) // tick_size
        ticks = []
        
        base_time = datetime(day.year, day.month, day.day, hour)
        
        for i in range(num_ticks):
            start = i * tick_size
            end = start + tick_size
            if end <= len(decompressed):
                # Unpack: milliseconds, ask, bid, ask_volume, bid_volume
                ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack('!IIIff', decompressed[start:end])
                
                # Convert timestamp
                tick_time = base_time + timedelta(milliseconds=ms)
                
                # Convert prices (divide by point value)
                point = 100000
                if symbol.lower() in ['usdrub', 'xagusd', 'xauusd']:
                    point = 1000
                
                ask_price = ask_raw / point
                bid_price = bid_raw / point
                
                # Convert volumes
                ask_volume = round(ask_vol * 1000000)
                bid_volume = round(bid_vol * 1000000)
                
                tick = TickData(tick_time, ask_price, bid_price, ask_volume, bid_volume)
                ticks.append(tick)
        
        return ticks
        
    except Exception as e:
        print(f"Error parsing {symbol} {day} hour {hour}: {e}")
        return []

def ticks_to_hourly_candle(ticks, hour_start):
    """Convert list of ticks to hourly candle"""
    if not ticks:
        return None
    
    # Use bid prices for candle (common practice)
    prices = [tick.bid for tick in ticks]
    volumes = [tick.bid_volume for tick in ticks]
    
    candle = HourlyCandle(
        timestamp=hour_start,
        open_price=prices[0],
        high=max(prices),
        low=min(prices),
        close_price=prices[-1],
        volume=sum(volumes)
    )
    
    return candle

async def download_symbol_year(session, symbol, year):
    """Download all data for one symbol for one year"""
    print(f"📥 Downloading {symbol} {year}...")
    start_time = time.time()
    
    # Generate all URLs for the year
    tasks = []
    url_to_date_hour = {}
    
    for month in range(12):
        for day in range(1, 32):
            try:
                test_date = date(year, month + 1, day)
                # Only download weekdays (forex market closed on weekends)
                if test_date.weekday() < 5:
                    for hour in range(24):
                        url = f"https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
                        task = asyncio.create_task(download_single_file(session, url))
                        tasks.append(task)
                        url_to_date_hour[url] = (test_date, hour)
            except ValueError:
                continue  # Invalid date (e.g., Feb 30)
    
    # Download all files in parallel
    results = await asyncio.gather(*tasks)
    
    # Process results
    daily_data = defaultdict(list)  # date -> list of hourly candles
    successful_downloads = 0
    
    for url, data in results:
        if data and len(data) > 0:
            day, hour = url_to_date_hour[url]
            
            # Parse ticks
            ticks = parse_bi5_data(data, symbol, day, hour)
            
            if ticks:
                # Convert to hourly candle
                hour_start = datetime(day.year, day.month, day.day, hour)
                candle = ticks_to_hourly_candle(ticks, hour_start)
                if candle:
                    daily_data[day].append(candle)
                    successful_downloads += 1
    
    elapsed = time.time() - start_time
    print(f"✅ {symbol} {year}: {successful_downloads} candles in {elapsed:.1f}s")
    
    return symbol, year, daily_data

def write_csv_data(symbol, year, daily_data, output_dir):
    """Write processed data to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{symbol}_H1_{year}.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Write data (sorted by date and hour)
        candle_count = 0
        for day in sorted(daily_data.keys()):
            # Sort candles by hour
            day_candles = sorted(daily_data[day], key=lambda c: c.timestamp)
            for candle in day_candles:
                writer.writerow([
                    candle.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    f"{candle.open:.5f}",
                    f"{candle.high:.5f}",
                    f"{candle.low:.5f}",
                    f"{candle.close:.5f}",
                    candle.volume
                ])
                candle_count += 1
    
    print(f"💾 Saved {candle_count} candles to {filename}")
    return candle_count

async def ultra_fast_download(symbols, start_year, end_year, output_dir=OUTPUT_DIR):
    """Main ultra-fast download function"""
    
    # Configure aiohttp for speed
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS,
        limit_per_host=MAX_CONCURRENT_REQUESTS,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout, 
        headers=headers
    ) as session:
        
        print(f"🚀 ULTRA-FAST DUKASCOPY DOWNLOADER")
        print(f"📈 Symbols: {', '.join(symbols)}")
        print(f"📅 Years: {start_year}-{end_year}")
        print(f"⚡ Max concurrent: {MAX_CONCURRENT_REQUESTS}")
        print(f"📁 Output: {output_dir}")
        print()
        
        # Download all symbols and years
        all_tasks = []
        for symbol in symbols:
            for year in range(start_year, end_year + 1):
                task = download_symbol_year(session, symbol, year)
                all_tasks.append(task)
        
        total_start = time.time()
        results = await asyncio.gather(*all_tasks)
        
        # Write CSV files
        print("\n💾 Writing CSV files...")
        total_candles = 0
        for symbol, year, daily_data in results:
            candle_count = write_csv_data(symbol, year, daily_data, output_dir)
            total_candles += candle_count
        
        total_time = time.time() - total_start
        
        print(f"\n🎉 ULTRA-FAST DOWNLOAD COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📊 Total candles: {total_candles}")
        print(f"⚡ Speed: {total_candles/total_time:.1f} candles/second")
        print(f"📁 Files saved to: {output_dir}")

if __name__ == "__main__":
    # Configuration - edit these as needed
    SYMBOLS = ["AUDCHF", "EURUSD", "GBPUSD"]
    START_YEAR = 2023
    END_YEAR = 2024
    
    print("Ultra-Fast Dukascopy Downloader")
    print("=" * 50)
    
    # Check if user wants to customize
    response = input(f"Download {', '.join(SYMBOLS)} from {START_YEAR}-{END_YEAR}? (y/n): ")
    if response.lower() != 'y':
        print("Edit the SYMBOLS, START_YEAR, and END_YEAR variables in the script")
        exit()
    
    # Run the download
    try:
        asyncio.run(ultra_fast_download(SYMBOLS, START_YEAR, END_YEAR))
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
