#!/usr/bin/env python3
"""
Ultra-fast Dukascopy downloader - aggressive optimization
"""

import asyncio
import aiohttp
import aiofiles
import time
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Ultra-aggressive settings
MAX_CONCURRENT_REQUESTS = 50  # Much higher concurrency
CHUNK_SIZE = 365  # Download year-sized chunks
TIMEOUT = 5  # Very fast timeout

async def download_year_batch(session, symbol, year, output_dir):
    """Download a full year of data in parallel"""
    print(f"Starting {symbol} {year}...")
    start_time = time.time()
    
    tasks = []
    for month in range(12):
        for day in range(1, 32):  # Will handle invalid dates gracefully
            try:
                test_date = date(year, month + 1, day)
                if test_date.weekday() < 5:  # Only weekdays
                    for hour in range(24):
                        url = f"https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
                        task = asyncio.create_task(download_single_file(session, url))
                        tasks.append(task)
            except ValueError:
                continue  # Invalid date
    
    # Execute all downloads for the year in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process successful downloads
    valid_data = [r for r in results if isinstance(r, bytes) and len(r) > 0]
    
    elapsed = time.time() - start_time
    print(f"✅ {symbol} {year}: {len(valid_data)} files in {elapsed:.1f}s")
    
    return valid_data

async def download_single_file(session, url):
    """Download a single bi5 file with minimal overhead"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as response:
            if response.status == 200:
                return await response.read()
            return b''
    except:
        return b''

async def ultra_fast_download(symbols, start_year, end_year):
    """Ultra-fast parallel download using aiohttp"""
    
    # Configure aiohttp for maximum speed
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
        
        print(f"🚀 Starting ultra-fast download for {len(symbols)} symbols")
        print(f"📅 Years: {start_year}-{end_year}")
        print(f"⚡ Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
        
        all_tasks = []
        for symbol in symbols:
            for year in range(start_year, end_year + 1):
                task = download_year_batch(session, symbol, year, "./data")
                all_tasks.append(task)
        
        # Download everything in parallel
        total_start = time.time()
        results = await asyncio.gather(*all_tasks)
        total_time = time.time() - total_start
        
        # Calculate stats
        total_files = sum(len(year_data) for year_data in results)
        total_symbols = len(symbols)
        total_years = end_year - start_year + 1
        
        print(f"\n🎉 ULTRA-FAST DOWNLOAD COMPLETE!")
        print(f"📊 Downloaded {total_files} files in {total_time:.1f} seconds")
        print(f"⚡ Speed: {total_files/total_time:.1f} files/second")
        print(f"📈 {total_symbols} symbols × {total_years} years")

def process_data_parallel(raw_data_chunks):
    """Process downloaded data in parallel using all CPU cores"""
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Process each chunk in parallel
        futures = [executor.submit(decompress_and_convert, chunk) for chunk in raw_data_chunks]
        results = [f.result() for f in futures]
    return results

def decompress_and_convert(data_chunk):
    """Decompress and convert data (runs in separate process)"""
    # Your existing decompression logic here
    pass

if __name__ == "__main__":
    # Example usage - download 5 years of data for multiple symbols
    symbols = ["AUDCHF", "EURUSD", "GBPUSD", "USDJPY"]
    
    # This should download MUCH faster than the current approach
    asyncio.run(ultra_fast_download(symbols, 2020, 2024))
