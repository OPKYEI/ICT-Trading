# src/data_processing/data_loader.py
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Union, List
from pathlib import Path
import glob

class DataLoader:
    """Load and preprocess financial data for ICT analysis"""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path('data/raw')
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, 
                  symbol: str, 
                  start_date: str, 
                  end_date: str, 
                  interval: str = '1h',
                  data_source: str = 'local') -> pd.DataFrame:
        """
        Load OHLCV data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d, 1wk, 1mo)
            data_source: 'local' or 'yfinance'
        
        Returns:
            DataFrame with OHLCV data
        """
        if data_source == 'local':
            df = self._load_local_data(symbol, start_date, end_date, interval)
            if df is not None and not df.empty:
                return df
            else:
                print(f"Local data not found for {symbol}, falling back to yfinance")
        
        # Fallback to yfinance
        return self._load_yfinance_data(symbol, start_date, end_date, interval)
    
    def _load_local_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data from local files"""
        # Look for files matching pattern: EURUSD_1h.csv, EURUSD_1h_2023.csv, etc.
        pattern = f"{symbol}*{interval}*.csv"
        files = glob.glob(str(self.data_path / pattern))
        
        if not files:
            return None
        
        # Load all matching files and combine
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not dfs:
            return None
        
        # Combine and filter by date range
        df = pd.concat(dfs).sort_index()
        df = df[start_date:end_date]
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in required_cols):
            return df[required_cols]
        
        return None
    
    def _load_yfinance_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Load data from yfinance"""
        try:
            # Convert symbol format if needed (EURUSD -> EURUSD=X)
            yf_symbol = symbol if '=' in symbol else f"{symbol}=X"
            
            df = yf.download(yf_symbol, start=start_date, end=end_date, interval=interval)
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
            
            return df[required_cols]
        
        except Exception as e:
            raise Exception(f"Error loading data from yfinance for {symbol}: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        # Check for missing values
        if df.isnull().any().any():
            return False
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] <= 0).any().any():
            return False
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            return False
        
        # Check high >= open, close
        if ((df['high'] < df['open']) | (df['high'] < df['close'])).any():
            return False
        
        # Check low <= open, close
        if ((df['low'] > df['open']) | (df['low'] > df['close'])).any():
            return False
        
        return True
    
    def resample_data(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to a different timeframe"""
        resample_map = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        return df.resample(target_timeframe).agg(resample_map).dropna()
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval: str, 
                  start_date: str = None, end_date: str = None):
        """Save data to local file"""
        if start_date and end_date:
            filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        else:
            filename = f"{symbol}_{interval}.csv"
        
        filepath = self.data_path / filename
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")