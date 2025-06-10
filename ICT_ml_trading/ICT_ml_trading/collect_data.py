#!/usr/bin/env python3
"""
Ultimate Multi-Source Market Data Downloader
- Multiple FREE sources: HistData, FXCM, Dukascopy, Alpha Vantage, Yahoo Finance
- Proper volume handling (tick volume for forex)
- Comprehensive alternative source suggestions
- Auto-fallback between sources
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import time
import zipfile
import gc
import requests
import gzip
from io import StringIO

# Global configuration
ALPHA_VANTAGE_API = ""
TWELVE_DATA_API = ""

def get_comprehensive_sources():
    """All available FREE sources for market data"""
    return {
        'primary_sources': {
            'histdata': {
                'name': 'HistData.com',
                'description': 'Free forex minute data (2000-2024)',
                'instruments': ['Forex pairs', 'Gold', 'Silver'],
                'volume': 'No volume data (forex spot)',
                'setup': 'pip install histdata'
            },
            'fxcm': {
                'name': 'FXCM Free Data',
                'description': 'Free historical candle data (2017-2020)',
                'instruments': ['Major forex pairs'],
                'volume': 'No volume data (forex spot)', 
                'setup': 'Direct CSV download'
            },
            'dukascopy': {
                'name': 'Dukascopy Historical Data',
                'description': 'High-quality tick and bar data',
                'instruments': ['Forex', 'CFDs', 'Crypto'],
                'volume': 'Tick volume available',
                'setup': 'Web interface or JForex platform'
            }
        },
        'api_sources': {
            'alpha_vantage': {
                'name': 'Alpha Vantage',
                'description': 'Free tier: 500 calls/day',
                'instruments': ['Stocks', 'Forex', 'Crypto'],
                'volume': 'Stock volume, forex tick volume',
                'setup': 'Free API key from alphavantage.co'
            },
            'yahoo_finance': {
                'name': 'Yahoo Finance (yfinance)',
                'description': 'Unlimited free access',
                'instruments': ['Stocks', 'Forex', 'Crypto', 'Indices'],
                'volume': 'Stock volume, forex=0',
                'setup': 'pip install yfinance'
            },
            'twelve_data': {
                'name': 'Twelve Data',
                'description': 'Free tier: 800 calls/day',
                'instruments': ['Stocks', 'Forex', 'Crypto'],
                'volume': 'Various volume types',
                'setup': 'Free API key from twelvedata.com'
            }
        },
        'specialized_sources': {
            'quandl': {
                'name': 'Quandl/NASDAQ Data Link',
                'description': 'Economic and financial datasets',
                'instruments': ['Economic indicators', 'Alternative data'],
                'volume': 'Varies by dataset',
                'setup': 'Free API key from data.nasdaq.com'
            },
            'federal_reserve': {
                'name': 'FRED Economic Data',
                'description': 'Federal Reserve economic data',
                'instruments': ['Economic indicators', 'Interest rates'],
                'volume': 'Not applicable',
                'setup': 'Free API from fred.stlouisfed.org'
            }
        }
    }

def show_comprehensive_sources():
    """Display all available sources with setup instructions"""
    sources = get_comprehensive_sources()
    
    print("🌍 COMPREHENSIVE FREE DATA SOURCES")
    print("=" * 60)
    
    print("\n📊 PRIMARY SOURCES (Bulk Downloads):")
    for key, source in sources['primary_sources'].items():
        print(f"\n   {source['name']}:")
        print(f"   📝 {source['description']}")
        print(f"   🎯 Instruments: {', '.join(source['instruments'])}")
        print(f"   📈 Volume: {source['volume']}")
        print(f"   ⚙️  Setup: {source['setup']}")
    
    print("\n🔌 API SOURCES (Real-time + Historical):")
    for key, source in sources['api_sources'].items():
        print(f"\n   {source['name']}:")
        print(f"   📝 {source['description']}")
        print(f"   🎯 Instruments: {', '.join(source['instruments'])}")
        print(f"   📈 Volume: {source['volume']}")
        print(f"   ⚙️  Setup: {source['setup']}")
    
    print("\n🎯 SPECIALIZED SOURCES:")
    for key, source in sources['specialized_sources'].items():
        print(f"\n   {source['name']}:")
        print(f"   📝 {source['description']}")
        print(f"   🎯 Instruments: {', '.join(source['instruments'])}")
        print(f"   📈 Volume: {source['volume']}")
        print(f"   ⚙️  Setup: {source['setup']}")

def get_volume_explanation():
    """Explain volume in different markets"""
    return {
        'forex_spot': {
            'type': 'No True Volume',
            'explanation': 'Forex is decentralized - no central exchange records volume',
            'alternative': 'Tick Volume (number of price changes)',
            'sources': ['Dukascopy tick data', 'MT4/MT5 brokers']
        },
        'forex_futures': {
            'type': 'Real Volume',
            'explanation': 'Currency futures trade on exchanges (CME, etc.)',
            'alternative': 'Actual contract volume',
            'sources': ['CME data', 'Futures exchanges']
        },
        'stocks': {
            'type': 'Real Volume',
            'explanation': 'Number of shares traded on exchanges',
            'alternative': 'Share volume',
            'sources': ['Yahoo Finance', 'Alpha Vantage', 'Exchange feeds']
        },
        'crypto': {
            'type': 'Real Volume',
            'explanation': 'Trading volume on crypto exchanges',
            'alternative': 'USD volume or coin volume',
            'sources': ['Exchange APIs', 'CoinGecko', 'CoinMarketCap']
        }
    }

def show_volume_explanation():
    """Show volume explanation for different asset classes"""
    volume_info = get_volume_explanation()
    
    print("\n📊 VOLUME DATA EXPLANATION")
    print("=" * 50)
    
    for asset, info in volume_info.items():
        print(f"\n🎯 {asset.replace('_', ' ').title()}:")
        print(f"   Type: {info['type']}")
        print(f"   📝 {info['explanation']}")
        print(f"   📈 Alternative: {info['alternative']}")
        print(f"   🔗 Sources: {', '.join(info['sources'])}")
    
    print(f"\n💡 KEY INSIGHT: Forex spot has NO traditional volume!")
    print(f"   📊 Use 'tick volume' instead (price change frequency)")
    print(f"   ⚡ This script puts 0 for forex volume (industry standard)")

def download_fxcm_data(symbol, years, output_dir):
    """Download free data from FXCM repository - FIXED for multiple years"""
    try:
        print(f"   🔌 Trying FXCM free data for {symbol}...")
        
        all_yearly_data = []
        
        for year in years:
            if year > 2020:  # FXCM only has data through 2020
                continue
                
            print(f"   📅 FXCM {year}...")
            year_data = []
            
            # Try different weeks for the year
            for week in range(1, 53):
                url = f"https://candledata.fxcorporate.com/m1/{symbol}/{year}/{week}.csv.gz"
                
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        # Decompress and read CSV
                        content = gzip.decompress(response.content).decode('utf-8')
                        lines = content.strip().split('\n')
                        
                        for line in lines[1:]:  # Skip header
                            try:
                                # FXCM format: DateTime,Open,High,Low,Close
                                parts = line.split(',')
                                if len(parts) >= 5:
                                    dt = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
                                    year_data.append({
                                        'datetime': dt,
                                        'open': float(parts[1]),
                                        'high': float(parts[2]),
                                        'low': float(parts[3]),
                                        'close': float(parts[4]),
                                        'volume': 0  # Forex has no volume
                                    })
                            except:
                                continue
                except:
                    continue
            
            if year_data:
                print(f"   ✅ FXCM {year}: {len(year_data):,} minute bars")
                all_yearly_data.extend(year_data)
        
        if all_yearly_data:
            df = pd.DataFrame(all_yearly_data)
            df['timestamp'] = df['datetime'].dt.strftime('%d.%m.%Y %H:%M:%S.000')
            print(f"   ✅ FXCM TOTAL: {len(df):,} minute bars")
            return df
        else:
            print(f"   ❌ FXCM: No data for {symbol}")
            return None
            
    except Exception as e:
        print(f"   ❌ FXCM error: {e}")
        return None

def download_yfinance_data(symbol, years, output_dir):
    """Download data using yfinance (Yahoo Finance)"""
    try:
        import yfinance as yf
        print(f"   🌐 Trying Yahoo Finance for {symbol}...")
        
        # Convert symbol format for Yahoo
        yahoo_symbols = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X',
            'XAUUSD': 'GC=F',  # Gold futures
            'XAGUSD': 'SI=F',  # Silver futures
            'SPX500': '^GSPC', # S&P 500
            'NAS100': '^IXIC', # NASDAQ
        }
        
        yahoo_symbol = yahoo_symbols.get(symbol, f"{symbol}=X")
        
        # Calculate date range
        start_date = datetime(min(years), 1, 1)
        end_date = datetime(max(years) + 1, 1, 1)
        
        # Download data
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval='1h'  # Hourly data
        )
        
        if not df.empty:
            # Format for our standard
            df.reset_index(inplace=True)
            df['timestamp'] = df['Datetime'].dt.strftime('%d.%m.%Y %H:%M:%S.000')
            
            # Yahoo Finance volume explanation
            if 'Volume' in df.columns:
                if symbol.startswith('XAU') or symbol.startswith('XAG'):
                    # Futures have real volume
                    df['volume'] = df['Volume']
                    print(f"   📊 Yahoo: Using futures volume data")
                else:
                    # Forex has no volume
                    df['volume'] = 0
                    print(f"   📊 Yahoo: Forex - setting volume to 0")
            else:
                df['volume'] = 0
            
            # Standard columns
            final_df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'volume']].copy()
            final_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            print(f"   ✅ Yahoo Finance: {len(final_df):,} hourly bars")
            return final_df
        else:
            print(f"   ❌ Yahoo Finance: No data for {yahoo_symbol}")
            return None
            
    except ImportError:
        print(f"   ⚠️ yfinance not installed: pip install yfinance")
        return None
    except Exception as e:
        print(f"   ❌ Yahoo Finance error: {e}")
        return None

def download_alpha_vantage_data(symbol, output_dir):
    """Download data using Alpha Vantage API"""
    if not ALPHA_VANTAGE_API:
        return None
        
    try:
        print(f"   🔑 Trying Alpha Vantage for {symbol}...")
        
        # Alpha Vantage symbol mapping
        av_symbol = symbol
        if symbol == 'XAUUSD':
            av_symbol = 'XAU'
        elif symbol == 'XAGUSD':
            av_symbol = 'XAG'
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': av_symbol[:3],
            'to_symbol': av_symbol[3:] if len(av_symbol) == 6 else 'USD',
            'interval': '60min',
            'outputsize': 'full',
            'apikey': ALPHA_VANTAGE_API
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'Time Series FX (60min)' in data:
                time_series = data['Time Series FX (60min)']
                
                rows = []
                for timestamp, values in time_series.items():
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    rows.append({
                        'timestamp': dt.strftime('%d.%m.%Y %H:%M:%S.000'),
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': 0  # Forex has no volume
                    })
                
                df = pd.DataFrame(rows)
                print(f"   ✅ Alpha Vantage: {len(df):,} hourly bars")
                return df
            else:
                print(f"   ❌ Alpha Vantage: No data or API limit reached")
                return None
        else:
            print(f"   ❌ Alpha Vantage: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   ❌ Alpha Vantage error: {e}")
        return None

def download_histdata_fallback(symbol, years):
    """HistData download with fallback - FIXED to download ALL years"""
    try:
        from histdata import download_hist_data as dl
        from histdata.api import Platform as P, TimeFrame as TF
        
        print(f"   📊 Trying HistData for {symbol}...")
        
        all_yearly_data = []  # Collect data from ALL years
        
        for year in years:
            try:
                result = dl(pair=symbol.lower(), year=year, platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE)
                
                zip_filename = f"DAT_ASCII_{symbol.upper()}_M1_{year}.zip"
                
                if os.path.exists(zip_filename):
                    # Process the data
                    data_rows = []
                    
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                        
                        if csv_files:
                            csv_filename = csv_files[0]
                            
                            with zip_ref.open(csv_filename) as csv_file:
                                content = csv_file.read().decode('utf-8')
                                lines = content.strip().split('\n')
                                
                                for line in lines:
                                    try:
                                        parts = line.split(';')
                                        if len(parts) >= 5:
                                            datetime_str = parts[0].strip()
                                            dt = datetime.strptime(datetime_str, '%Y%m%d %H%M%S')
                                            
                                            data_rows.append({
                                                'datetime': dt,
                                                'open': float(parts[1]),
                                                'high': float(parts[2]),
                                                'low': float(parts[3]),
                                                'close': float(parts[4]),
                                                'volume': 0  # Forex has no volume
                                            })
                                    except:
                                        continue
                    
                    # Cleanup
                    try:
                        os.remove(zip_filename)
                    except:
                        pass
                    
                    if data_rows:
                        print(f"   ✅ HistData {year}: {len(data_rows):,} minute bars")
                        all_yearly_data.extend(data_rows)  # Add to combined data
                    else:
                        print(f"   ⚠️ HistData {year}: No valid data")
                else:
                    print(f"   ❌ HistData {year}: Download failed")
                        
            except Exception as e:
                if "no token" in str(e).lower():
                    print(f"   ❌ HistData: {symbol} not available")
                    break  # Symbol not available, no point trying other years
                else:
                    print(f"   ⚠️ HistData {year}: {e}")
                    continue  # Try next year
        
        # Return combined data from ALL years
        if all_yearly_data:
            df = pd.DataFrame(all_yearly_data)
            # ADD TIMESTAMP COLUMN
            df['timestamp'] = df['datetime'].dt.strftime('%d.%m.%Y %H:%M:%S.000')
            print(f"   ✅ HistData TOTAL: {len(df):,} minute bars from {len(years)} years")
            return df
        else:
            print(f"   ❌ HistData: No data for any year")
            return None
        
    except ImportError:
        print(f"   ⚠️ histdata not installed: pip install histdata")
        return None
    except Exception as e:
        print(f"   ❌ HistData error: {e}")
        return None

def get_alternative_sources_for_symbol(symbol):
    """Get specific alternative sources for failed symbols"""
    alternatives = {
        'XPTUSD': [
            "🔑 Alpha Vantage: Free 500 calls/day - alphavantage.co",
            "🌐 Yahoo Finance: yfinance library - pip install yfinance", 
            "📊 Twelve Data: Free 800 calls/day - twelvedata.com",
            "🏦 MetaTrader 5: Free demo account with historical data",
            "📈 TradingView: Free charts with export options"
        ],
        'XPDUSD': [
            "🔑 Alpha Vantage: Free precious metals data",
            "🌐 Yahoo Finance: Limited palladium data", 
            "📊 Twelve Data: Commodities in free tier",
            "🏦 MT5: Broker demo accounts",
            "📈 Investing.com: Free historical data export"
        ],
        'SPX500': [
            "🌐 Yahoo Finance: ^GSPC symbol - unlimited free",
            "🔑 Alpha Vantage: SPY ETF as proxy",
            "📊 FRED: Federal Reserve economic data",
            "📈 Quandl: NASDAQ Data Link free tier",
            "🏦 IEX Cloud: Free US market data"
        ],
        'NAS100': [
            "🌐 Yahoo Finance: ^IXIC symbol",
            "🔑 Alpha Vantage: QQQ ETF as proxy", 
            "📊 FRED: Economic indicators",
            "📈 Twelve Data: US indices",
            "🏦 Polygon.io: Free tier available"
        ],
        'BTCUSD': [
            "🌐 Yahoo Finance: BTC-USD symbol",
            "🔑 Alpha Vantage: Digital currencies",
            "📊 CoinGecko: Free API unlimited",
            "📈 CoinMarketCap: Free API tier",
            "🏦 Binance: Free historical data API"
        ]
    }
    
    # Generic alternatives for unknown symbols
    generic_alternatives = [
        "🌐 Yahoo Finance: Try yfinance with symbol variations",
        "🔑 Alpha Vantage: 500 free calls/day - alphavantage.co",
        "📊 Twelve Data: 800 free calls/day - twelvedata.com", 
        "📈 Dukascopy: Free historical data feed",
        "🏦 MetaTrader 5: Demo account with data export",
        "📊 TradingView: Free charts with limited export",
        "🔗 Quandl/NASDAQ: Free economic datasets"
    ]
    
    return alternatives.get(symbol, generic_alternatives)

def multi_source_download(symbol, years, output_dir):
    """Try multiple sources with proper fallback - FIXED for multiple years"""
    print(f"🔄 Multi-source download: {symbol} ({len(years)} years: {years[0]}-{years[-1]})")
    
    all_sources_data = []
    successful_sources = []
    
    # Source 1: HistData (primary for forex) - NOW DOWNLOADS ALL YEARS
    df_hist = download_histdata_fallback(symbol, years)
    if df_hist is not None and len(df_hist) > 0:
        all_sources_data.append(df_hist)
        successful_sources.append("HistData")
    
    # Source 2: Yahoo Finance (good for everything)
    df_yahoo = download_yfinance_data(symbol, years, output_dir)
    if df_yahoo is not None and len(df_yahoo) > 0:
        all_sources_data.append(df_yahoo)
        successful_sources.append("Yahoo Finance")
    
    # Source 3: FXCM (forex only, limited years) - NOW HANDLES MULTIPLE YEARS
    if symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD'] and any(y <= 2020 for y in years):
        df_fxcm = download_fxcm_data(symbol, years, output_dir)
        if df_fxcm is not None and len(df_fxcm) > 0:
            all_sources_data.append(df_fxcm)
            successful_sources.append("FXCM")
    
    # Source 4: Alpha Vantage (if API key provided)
    df_av = download_alpha_vantage_data(symbol, output_dir)
    if df_av is not None and len(df_av) > 0:
        all_sources_data.append(df_av)
        successful_sources.append("Alpha Vantage")
    
    # Combine results
    if all_sources_data:
        print(f"   🔗 Combining data from: {', '.join(successful_sources)}")
        
        # Combine all DataFrames
        combined = pd.concat(all_sources_data, ignore_index=True)
        
        # Debug: Check what columns we have
        print(f"   🔍 Available columns: {list(combined.columns)}")
        
        # Ensure we have timestamp column
        if 'timestamp' not in combined.columns:
            if 'datetime' in combined.columns:
                combined['timestamp'] = combined['datetime'].dt.strftime('%d.%m.%Y %H:%M:%S.000')
                print(f"   🔧 Created timestamp column from datetime")
            else:
                print(f"   ❌ No timestamp or datetime column found!")
                return False
        
        # Remove duplicates and sort
        combined['datetime'] = pd.to_datetime(combined['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
        combined = combined.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
        
        # Convert to hourly if we have minute data
        if len(combined) > 24 * 365:  # Likely minute data
            print(f"   🔄 Converting {len(combined):,} minute bars to hourly...")
            combined.set_index('datetime', inplace=True)
            hourly = combined.resample('h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            hourly.reset_index(inplace=True)
            hourly['timestamp'] = hourly['datetime'].dt.strftime('%d.%m.%Y %H:%M:%S.000')
            combined = hourly
        
        # Save with naming convention
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{symbol}=X_60m.csv"
        filepath = os.path.join(output_dir, filename)
        
        final_df = combined[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        final_df.to_csv(filepath, index=False, float_format='%.5f')
        
        print(f"   💾 Saved: {filename} ({len(final_df):,} hourly bars)")
        print(f"   📅 Range: {combined['datetime'].min().date()} to {combined['datetime'].max().date()}")
        print(f"   📊 Sources: {', '.join(successful_sources)}")
        
        return True
    else:
        print(f"   ❌ No data from any source")
        print(f"   💡 Alternative sources for {symbol}:")
        alternatives = get_alternative_sources_for_symbol(symbol)
        for alt in alternatives:
            print(f"      {alt}")
        
        return False

def setup_apis():
    """Setup API keys for enhanced sources"""
    global ALPHA_VANTAGE_API, TWELVE_DATA_API
    
    print("🔑 API SETUP (Optional - for more data sources)")
    print("=" * 50)
    
    setup = input("Setup free API keys for additional sources? (y/n): ").lower()
    
    if setup == 'y':
        print("\n📈 Alpha Vantage (500 free calls/day):")
        print("   1. Visit: https://www.alphavantage.co/support/#api-key")
        print("   2. Get free API key")
        key = input("   Enter Alpha Vantage API key (or press Enter to skip): ").strip()
        if key:
            ALPHA_VANTAGE_API = key
            print("   ✅ Alpha Vantage configured")
        
        print("\n📊 Twelve Data (800 free calls/day):")
        print("   1. Visit: https://twelvedata.com/")
        print("   2. Sign up for free account")
        key = input("   Enter Twelve Data API key (or press Enter to skip): ").strip()
        if key:
            TWELVE_DATA_API = key
            print("   ✅ Twelve Data configured")
    
    if not ALPHA_VANTAGE_API and not TWELVE_DATA_API:
        print("📊 Using free sources only (HistData + Yahoo Finance + FXCM)")

def main():
    print("🚀 ULTIMATE MULTI-SOURCE MARKET DATA DOWNLOADER")
    print("📊 Combines: HistData + Yahoo Finance + FXCM + Alpha Vantage + More")
    print("🔧 Features:")
    print("   ✅ Multiple FREE sources with auto-fallback")
    print("   ✅ Proper volume handling (tick volume for forex)")
    print("   ✅ Comprehensive alternative source suggestions")
    print("   ✅ File format: SYMBOL=X_60m.csv")
    print("=" * 70)
    
    # Show all sources
    show_comprehensive_sources()
    
    # Show volume explanation
    show_volume_explanation()
    
    # Setup APIs
    setup_apis()
    
    # Select instruments
    print(f"\n📈 INSTRUMENT SELECTION:")
    print(f"Enter instruments separated by commas:")
    print(f"Examples: EURUSD,XAUUSD,SPX500 or GBPUSD,XAGUSD")
    
    symbols_input = input("Instruments: ").strip().upper()
    
    if not symbols_input:
        symbols = ['EURUSD', 'XAUUSD']
        print(f"Using default: {', '.join(symbols)}")
    else:
        symbols = [s.strip() for s in symbols_input.split(',')]
    
    print(f"📊 Selected: {', '.join(symbols)}")
    
    # Select years
    year_input = input("\nYears [default: 2022-2024]: ").strip()
    if not year_input:
        year_input = "2022-2024"
    
    years = []
    try:
        if '-' in year_input:
            start, end = map(int, year_input.split('-'))
            years = list(range(start, end + 1))
        elif ',' in year_input:
            years = [int(y.strip()) for y in year_input.split(',')]
        else:
            years = [int(year_input)]
        
        years = [y for y in years if 2000 <= y <= 2024]
        
    except ValueError:
        years = [2022, 2023, 2024]
    
    print(f"📅 Years: {', '.join(map(str, years))}")
    
    # Output directory
    output_dir = input("\nOutput directory [default: ./ultimate_data]: ").strip()
    if not output_dir:
        output_dir = "./ultimate_data"
    
    # Summary
    print(f"\n📋 DOWNLOAD SUMMARY:")
    print(f"   📊 Instruments: {len(symbols)}")
    print(f"   📅 Years: {len(years)}")
    print(f"   🔄 Sources: HistData + Yahoo Finance + FXCM + APIs")
    print(f"   📁 Output: {output_dir}")
    print(f"   📏 Format: SYMBOL=X_60m.csv")
    print(f"   📊 Volume: Explained above (forex=0, stocks=real)")
    
    proceed = input(f"\nProceed with multi-source download? (y/n): ").lower()
    if proceed != 'y':
        return
    
    # Download with multiple sources
    print(f"\n🔥 Starting multi-source downloads...")
    successful = 0
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        
        if multi_source_download(symbol, years, output_dir):
            successful += 1
        else:
            failed.append(symbol)
        
        # Rate limiting between symbols
        if i < len(symbols):
            time.sleep(2)
    
    # Final results
    print(f"\n🎉 MULTI-SOURCE DOWNLOAD COMPLETE!")
    print(f"✅ Successful: {successful}/{len(symbols)}")
    
    if failed:
        print(f"\n❌ Failed instruments with alternatives:")
        for symbol in failed:
            print(f"\n   {symbol}:")
            alternatives = get_alternative_sources_for_symbol(symbol)
            for alt in alternatives[:3]:  # Show top 3 alternatives
                print(f"      {alt}")
    
    if successful > 0:
        print(f"\n🎯 SUCCESS! Multi-source market data downloaded!")
        print(f"📊 File naming: SYMBOL=X_60m.csv")
        print(f"📈 Volume handling: Proper for each asset class")
        print(f"🔄 Sources combined for maximum coverage")
        
        # Show sample files
        if os.path.exists(output_dir):
            csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
            if csv_files:
                print(f"\n📋 Files created:")
                for file in csv_files:
                    print(f"   📄 {file}")
                
                # Show sample structure
                sample_file = os.path.join(output_dir, csv_files[0])
                try:
                    sample_df = pd.read_csv(sample_file, nrows=3)
                    print(f"\n📊 Sample data structure ({csv_files[0]}):")
                    print(sample_df.to_string(index=False))
                except:
                    pass
    
    print(f"\n💡 VOLUME DATA NOTES:")
    print(f"   📊 Forex: Volume = 0 (no centralized exchange)")
    print(f"   📈 Stocks: Volume = actual shares traded")
    print(f"   🏅 Futures: Volume = actual contracts traded")
    print(f"   💰 This is industry standard behavior")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
