# src/features/time_features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time, timedelta
import pytz

class TimeFeatures:
    """
    Handles ICT time-based features including:
    - Kill zones
    - Trading sessions
    - Day of week analysis
    - Time of day patterns
    - Midnight opens
    - Power of 3 (Accumulation, Manipulation, Distribution)
    """
    
    def __init__(self, timezone: str = 'America/New_York'):
        """
        Args:
            timezone: Timezone for time calculations (default: New York)
        """
        self.timezone = pytz.timezone(timezone)
        
        # Define ICT kill zones (New York time)
        self.kill_zones = {
            'asian_killzone': {'start': time(19, 0), 'end': time(0, 0)},  # 7 PM - 12 AM
            'london_open_killzone': {'start': time(2, 0), 'end': time(5, 0)},  # 2 AM - 5 AM
            'new_york_open_killzone': {'start': time(7, 0), 'end': time(10, 0)},  # 7 AM - 10 AM
            'london_close_killzone': {'start': time(10, 0), 'end': time(12, 0)},  # 10 AM - 12 PM
            'new_york_pm_killzone': {'start': time(13, 30), 'end': time(16, 0)}  # 1:30 PM - 4 PM
        }
        
        # Define trading sessions
        self.sessions = {
            'asian': {'start': time(19, 0), 'end': time(3, 0)},  # 7 PM - 3 AM
            'london': {'start': time(2, 0), 'end': time(11, 0)},  # 2 AM - 11 AM
            'new_york': {'start': time(7, 0), 'end': time(16, 0)},  # 7 AM - 4 PM
            'overlap': {'start': time(7, 0), 'end': time(11, 0)}  # 7 AM - 11 AM (London/NY overlap)
        }
        
        # ICT specific times
        self.ict_times = {
            'midnight_open': time(0, 0),
            'london_open': time(3, 0),
            'new_york_open': time(8, 30),  # NYSE open
            'london_close': time(11, 0),
            'new_york_close': time(16, 0),
            'cme_open': time(8, 20),
            'bond_close': time(15, 0)
        }
        
        # Day of week patterns
        self.day_patterns = {
            'monday': 'consolidation_or_false_move',
            'tuesday': 'potential_weekly_low_high',
            'wednesday': 'trend_continuation_or_reversal',  
            'thursday': 'reversal_day',
            'friday': 'profit_taking_or_squeeze'
        }
    
    def is_in_killzone(self, timestamp: pd.Timestamp, killzone_name: str) -> bool:
        """Check if timestamp is within a specific kill zone"""
        if killzone_name not in self.kill_zones:
            raise ValueError(f"Unknown kill zone: {killzone_name}")
        
        # Convert to NY time
        ny_time = timestamp.tz_convert(self.timezone).time()
        
        zone = self.kill_zones[killzone_name]
        start, end = zone['start'], zone['end']
        
        # Handle overnight kill zones
        if start > end:
            return ny_time >= start or ny_time <= end
        else:
            return start <= ny_time <= end
    
    def get_active_killzones(self, timestamp: pd.Timestamp) -> List[str]:
        """Get all active kill zones for a timestamp"""
        active_zones = []
        for zone_name in self.kill_zones:
            if self.is_in_killzone(timestamp, zone_name):
                active_zones.append(zone_name)
        return active_zones
    
    def is_in_session(self, timestamp: pd.Timestamp, session_name: str) -> bool:
        """Check if timestamp is within a specific trading session"""
        if session_name not in self.sessions:
            raise ValueError(f"Unknown session: {session_name}")
        
        # Convert to NY time
        ny_time = timestamp.tz_convert(self.timezone).time()
        
        session = self.sessions[session_name]
        start, end = session['start'], session['end']
        
        # Handle overnight sessions
        if start > end:
            return ny_time >= start or ny_time <= end
        else:
            return start <= ny_time <= end
    
    def get_active_sessions(self, timestamp: pd.Timestamp) -> List[str]:
        """Get all active sessions for a timestamp"""
        active_sessions = []
        for session_name in self.sessions:
            if self.is_in_session(timestamp, session_name):
                active_sessions.append(session_name)
        return active_sessions
    
    def get_day_of_week_bias(self, timestamp: pd.Timestamp) -> Dict[str, any]:
        """Get ICT day of week bias"""
        day_name = timestamp.strftime('%A').lower()
        
        bias = {
            'day': day_name,
            'day_number': timestamp.weekday(),  # 0 = Monday, 6 = Sunday
            'pattern': self.day_patterns.get(day_name, 'unknown'),
            'is_weekend': timestamp.weekday() >= 5
        }
        
        # Add ICT specific biases
        if day_name == 'monday':
            bias['bias'] = 'range_establishment'
        elif day_name == 'tuesday':
            bias['bias'] = 'weekly_low_or_high'
        elif day_name == 'wednesday':
            bias['bias'] = 'midweek_reversal_or_continuation'
        elif day_name == 'thursday':
            bias['bias'] = 'reversal_day'
        elif day_name == 'friday':
            bias['bias'] = 'profit_taking'
        else:
            bias['bias'] = 'weekend'
        
        return bias
    
    def get_time_of_day_features(self, timestamp: pd.Timestamp) -> Dict[str, any]:
        """Get comprehensive time features for a timestamp"""
        # Convert to NY time
        ny_timestamp = timestamp.tz_convert(self.timezone)
        ny_time = ny_timestamp.time()
        
        features = {
            'timestamp': timestamp,
            'ny_time': ny_time,
            'hour': ny_time.hour,
            'minute': ny_time.minute,
            'is_top_of_hour': ny_time.minute == 0,
            'is_30min_mark': ny_time.minute == 30,
            'active_killzones': self.get_active_killzones(timestamp),
            'active_sessions': self.get_active_sessions(timestamp),
            'day_of_week': self.get_day_of_week_bias(timestamp)
        }
        
        # Check for specific ICT times
        for ict_time_name, ict_time in self.ict_times.items():
            features[f'is_{ict_time_name}'] = (
                ny_time.hour == ict_time.hour and 
                abs(ny_time.minute - ict_time.minute) <= 5
            )
        
        # Power of 3 analysis
        features['power_of_3'] = self._get_power_of_3_phase(ny_time)
        
        return features
    
    def _get_power_of_3_phase(self, ny_time: time) -> str:
        """Determine Power of 3 phase based on time"""
        hour = ny_time.hour
        
        if 0 <= hour < 7:  # Asian session + early London
            return 'accumulation'
        elif 7 <= hour < 10:  # NY Open
            return 'manipulation'
        elif 10 <= hour < 16:  # NY Session
            return 'distribution'
        else:  # Evening
            return 'accumulation'
    
    def is_optimal_trading_time(self, timestamp: pd.Timestamp) -> Dict[str, bool]:
        """Check if timestamp is during optimal ICT trading times"""
        features = self.get_time_of_day_features(timestamp)
        
        optimal = {
            'is_kill_zone': bool(features['active_killzones']),
            'is_session_overlap': 'overlap' in features['active_sessions'],
            'is_major_session': bool(set(['london', 'new_york']) & set(features['active_sessions'])),
            'is_optimal_day': features['day_of_week']['day_number'] in [1, 2, 3],  # Tue, Wed, Thu
            'is_news_time': self._is_typical_news_time(features['ny_time'])
        }
        
        optimal['overall_optimal'] = (
            optimal['is_kill_zone'] and 
            optimal['is_major_session'] and 
            optimal['is_optimal_day']
        )
        
        return optimal
    
    def _is_typical_news_time(self, ny_time: time) -> bool:
        """Check if time is typical for economic news releases"""
        news_times = [
            time(8, 30),   # Major US economic data
            time(10, 0),   # More US data
            time(14, 0),   # FOMC minutes
            time(2, 0),    # European data
            time(4, 30)    # UK data
        ]
        
        for news_time in news_times:
            if (ny_time.hour == news_time.hour and 
                abs(ny_time.minute - news_time.minute) <= 15):
                return True
        
        return False
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time features to a DataFrame"""
        df = df.copy()
        
        # Ensure index is timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Basic time features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df.index.dayofweek >= 5
        
        # Kill zones
        for kz_name in self.kill_zones:
            df[f'in_{kz_name}'] = df.index.map(
                lambda x: self.is_in_killzone(x, kz_name)
            )
        
        # Sessions
        for session_name in self.sessions:
            df[f'in_{session_name}_session'] = df.index.map(
                lambda x: self.is_in_session(x, session_name)
            )
        
        # ICT specific times
        for time_name in self.ict_times:
            df[f'near_{time_name}'] = df.index.map(
                lambda x: self.get_time_of_day_features(x)[f'is_{time_name}']
            )
        
        # Day of week patterns
        df['day_pattern'] = df.index.map(
            lambda x: self.get_day_of_week_bias(x)['pattern']
        )
        
        # Power of 3
        df['power_of_3_phase'] = df.index.map(
            lambda x: self._get_power_of_3_phase(
                x.tz_convert(self.timezone).time()
            )
        )
        
        # Optimal trading times
        optimal_features = df.index.map(
            lambda x: self.is_optimal_trading_time(x)
        )
        df['is_optimal_time'] = [f['overall_optimal'] for f in optimal_features]
        
        return df