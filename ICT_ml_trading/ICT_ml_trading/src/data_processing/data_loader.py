# src/data_processing/data_loader.py
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Union, List
from pathlib import Path
import glob
from src.data_processing.timeframe_manager import TimeframeManager
class DataLoader:
    """Load and preprocess financial data for ICT analysis"""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path('data')
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, symbol: str, start_date: str, end_date: str, interval: str, data_source: str = 'local') -> pd.DataFrame:
        """Main data loading method with proper fallback."""

        df = None
        if data_source == 'local':
            try:
                df = self._load_local_data(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    print(f"Loaded local data for {symbol}_{interval}")
                    return df
                else:
                    print(f"No local data found for {symbol}_{interval}, falling back to yfinance.")
            except Exception as e:
                print(f"Local data loading error: {e}. Falling back to yfinance.")

        # fallback to yfinance explicitly
        try:
            return self._load_yfinance_data(symbol, start_date, end_date, interval)
        except Exception as e:
            raise Exception(f"Error loading data from yfinance: {e}")


    def _load_local_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        pattern = f"{symbol}*{interval}*.csv"
        files = glob.glob(str(self.data_path / pattern))

        if not files:
            raise FileNotFoundError(f"No local files found matching pattern {pattern} in {self.data_path.resolve()}")

        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                if 'timestamp' not in df.columns:
                    raise KeyError(f"'timestamp' column missing in file {file}")
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp']).set_index('timestamp')
                dfs.append(df)
            except Exception as e:
                raise Exception(f"Error loading file {file}: {e}")

        combined_df = pd.concat(dfs).sort_index()

        combined_df.columns = [col.lower() for col in combined_df.columns]

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in combined_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return combined_df[required_cols]


    def _load_yfinance_data(self, symbol, start_date, end_date, interval):
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "60m": "60m", "1h": "60m", "1d": "1d", "1wk": "1wk"
        }

        yf_interval = interval_map.get(interval.lower())
        if not yf_interval:
            raise ValueError(f"Invalid interval '{interval}' provided.")

        ticker = symbol if "=" in symbol else f"{symbol}=X"
        try:
            data = yf.download(tickers=ticker, start=start_date, end=end_date, interval=yf_interval, progress=True)

            if data.empty:
                raise ValueError(f"No data found for {ticker}")

            # Handling MultiIndex columns explicitly and accurately
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip().lower() for col in data.columns]
                # Rename explicitly to standard names
                rename_dict = {}
                for col in data.columns:
                    if 'open' in col:
                        rename_dict[col] = 'open'
                    elif 'high' in col:
                        rename_dict[col] = 'high'
                    elif 'low' in col:
                        rename_dict[col] = 'low'
                    elif 'close' in col:
                        rename_dict[col] = 'close'
                    elif 'volume' in col:
                        rename_dict[col] = 'volume'
                data = data.rename(columns=rename_dict)

            else:
                data.columns = data.columns.str.lower()

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing expected columns after processing: {missing_cols}")

            return data[required_cols]

        except Exception as e:
            raise Exception(f"Error loading data from yfinance for {ticker}: {str(e)}")

  
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
        
    def load_multi_timeframe(
        self,
        csv_path: Path,
        base_tf: str,
        extra_tfs: list[str]
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Load base timeframe CSV then generate and align extra timeframes.
        Returns (base_df, {tf: df, …}).
        """
        # 1) Load the base data
        base_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # ENSURE the index is datetime
        base_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # ensure a DatetimeIndex, interpreting day.month.year
        base_df.index = pd.to_datetime(
            base_df.index,
            dayfirst=True,
            errors="coerce"
        )

        # 2) Build MTF dict
        tfm = TimeframeManager()
        mtf = tfm.create_mtf_dataset(base_df, base_tf, extra_tfs)

        # 3) Align to common range
        aligned = tfm.align_timeframes(mtf)

        # 4) Pop the base and return extras
        extras = {tf: aligned[tf] for tf in extra_tfs}
        return aligned[base_tf], extras
