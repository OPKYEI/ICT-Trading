# src/data_processing/timeframe_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class TimeframeManager:
    """
    Manages multiple timeframes and synchronizes data for ICT analysis.
    Handles timeframe relationships and data alignment.
    """
    
    def __init__(self):
        self.timeframe_hierarchy = {
            '1M': {'minutes': 43200, 'parent': None},
            '1W': {'minutes': 10080, 'parent': '1M'},
            '1D': {'minutes': 1440, 'parent': '1W'},
            '4H': {'minutes': 240, 'parent': '1D'},
            '1H': {'minutes': 60, 'parent': '4H'},
            '30M': {'minutes': 30, 'parent': '1H'},
            '15M': {'minutes': 15, 'parent': '30M'},
            '5M': {'minutes': 5, 'parent': '15M'},
            '1M': {'minutes': 1, 'parent': '5M'}
        }
        
        self.timeframe_map = {
            '1M': 'M',   # Monthly
            '1W': 'W',   # Weekly
            '1D': 'D',   # Daily
            '4H': '4H',  # 4 Hours
            '1H': 'H',   # Hourly
            '30M': '30T', # 30 Minutes
            '15M': '15T', # 15 Minutes
            '5M': '5T',   # 5 Minutes
            '1M': 'T'     # 1 Minute
        }
    
    def get_parent_timeframe(self, timeframe: str) -> Optional[str]:
        """Get the parent timeframe for a given timeframe"""
        return self.timeframe_hierarchy.get(timeframe, {}).get('parent')
    
    def get_child_timeframes(self, timeframe: str) -> List[str]:
        """Get all child timeframes for a given timeframe"""
        children = []
        for tf, info in self.timeframe_hierarchy.items():
            if info.get('parent') == timeframe:
                children.append(tf)
        return children
    
    def align_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple timeframes to ensure proper synchronization
        
        Args:
            data_dict: Dictionary with timeframe as key and DataFrame as value
            
        Returns:
            Dictionary of aligned DataFrames
        """
        if not data_dict:
            return {}
        
        # Find common date range
        min_date = max(df.index.min() for df in data_dict.values())
        max_date = min(df.index.max() for df in data_dict.values())
        
        # Align all dataframes to common date range
        aligned_data = {}
        for timeframe, df in data_dict.items():
            aligned_df = df[min_date:max_date].copy()
            aligned_data[timeframe] = aligned_df
        
        return aligned_data
    
    def resample_to_higher_timeframe(self, df: pd.DataFrame, 
                                   from_tf: str, to_tf: str) -> pd.DataFrame:
        """
        Resample data from lower to higher timeframe
        
        Args:
            df: Source DataFrame
            from_tf: Source timeframe (e.g., '5M')
            to_tf: Target timeframe (e.g., '1H')
            
        Returns:
            Resampled DataFrame
        """
        if from_tf not in self.timeframe_map or to_tf not in self.timeframe_map:
            raise ValueError(f"Invalid timeframe. Use one of: {list(self.timeframe_map.keys())}")
        
        # Check if valid conversion (can only go up)
        from_minutes = self.timeframe_hierarchy[from_tf]['minutes']
        to_minutes = self.timeframe_hierarchy[to_tf]['minutes']
        
        if from_minutes >= to_minutes:
            raise ValueError(f"Can only resample from lower to higher timeframe. {from_tf} -> {to_tf} is invalid.")
        
        # Resample
        resampled = df.resample(self.timeframe_map[to_tf]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def get_higher_timeframe_candle(self, timestamp: pd.Timestamp, 
                                  lower_tf_data: pd.DataFrame, 
                                  higher_tf: str) -> pd.Series:
        """
        Get the higher timeframe candle that contains the given timestamp
        
        Args:
            timestamp: Timestamp to find the corresponding candle
            lower_tf_data: Lower timeframe data
            higher_tf: Higher timeframe to find the candle
            
        Returns:
            Higher timeframe candle data
        """
        # Resample to higher timeframe
        higher_tf_data = self.resample_to_higher_timeframe(
            lower_tf_data, 
            self._infer_timeframe(lower_tf_data), 
            higher_tf
        )
        
        # Find the candle containing the timestamp
        for idx, candle in higher_tf_data.iterrows():
            if idx <= timestamp < idx + pd.Timedelta(minutes=self.timeframe_hierarchy[higher_tf]['minutes']):
                return candle
        
        raise ValueError(f"No candle found for timestamp {timestamp}")
    
    def create_mtf_dataset(self, base_data: pd.DataFrame, 
                          base_tf: str, 
                          timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Create a multi-timeframe dataset from base data
        
        Args:
            base_data: Base timeframe data
            base_tf: Base timeframe
            timeframes: List of timeframes to generate
            
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        mtf_data = {base_tf: base_data.copy()}
        
        for tf in timeframes:
            if tf == base_tf:
                continue
            
            if self.timeframe_hierarchy[tf]['minutes'] > self.timeframe_hierarchy[base_tf]['minutes']:
                # Resample to higher timeframe
                mtf_data[tf] = self.resample_to_higher_timeframe(base_data, base_tf, tf)
            else:
                raise ValueError(f"Cannot resample from {base_tf} to {tf}. Can only resample to higher timeframes.")
        
        return mtf_data
    
    def _infer_timeframe(self, df: pd.DataFrame) -> str:
        """Infer timeframe from DataFrame index"""
        if len(df) < 2:
            raise ValueError("Need at least 2 rows to infer timeframe")
        
        # Calculate average time difference
        time_diff = df.index[1:] - df.index[:-1]
        avg_minutes = time_diff.mean().total_seconds() / 60
        
        # Find closest matching timeframe
        closest_tf = None
        min_diff = float('inf')
        
        for tf, info in self.timeframe_hierarchy.items():
            diff = abs(info['minutes'] - avg_minutes)
            if diff < min_diff:
                min_diff = diff
                closest_tf = tf
        
        return closest_tf
    
    def get_session_boundaries(self, date: pd.Timestamp, session: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get session start and end times for a given date
        
        Args:
            date: Date to get session boundaries
            session: Session name ('asian', 'london', 'new_york')
            
        Returns:
            Tuple of (start_time, end_time)
        """
        sessions = {
            'asian': (0, 9),      # 00:00 - 09:00 GMT
            'london': (7, 16),    # 07:00 - 16:00 GMT  
            'new_york': (12, 21)  # 12:00 - 21:00 GMT
        }
        
        if session not in sessions:
            raise ValueError(f"Invalid session. Use one of: {list(sessions.keys())}")
        
        start_hour, end_hour = sessions[session]
        
        start_time = date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_time = date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        return start_time, end_time
    
    def get_killzone_times(self, date: pd.Timestamp) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Get ICT kill zone times for a given date
        
        Args:
            date: Date to get kill zones
            
        Returns:
            Dictionary with kill zone names and (start, end) times
        """
        killzones = {
            'london_open': (7, 9),      # 07:00 - 09:00 GMT
            'new_york_open': (12, 14),  # 12:00 - 14:00 GMT
            'london_close': (14, 16),   # 14:00 - 16:00 GMT
            'new_york_close': (19, 21)  # 19:00 - 21:00 GMT
        }
        
        result = {}
        for name, (start_hour, end_hour) in killzones.items():
            start_time = date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            end_time = date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            result[name] = (start_time, end_time)
        
        return result