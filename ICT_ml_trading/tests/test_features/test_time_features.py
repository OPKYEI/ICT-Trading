# tests/test_features/test_time_features.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from src.features.time_features import TimeFeatures

class TestTimeFeatures:
    
    @pytest.fixture
    def time_features(self):
        """Create TimeFeatures instance"""
        return TimeFeatures(timezone='America/New_York')
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with various timestamps"""
        dates = pd.date_range(
            start='2023-01-02 00:00:00',  # Monday
            end='2023-01-08 23:00:00',    # Sunday
            freq='1H',
            tz='UTC'
        )
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        return data
    
    def test_is_in_killzone(self, time_features):
        """Test kill zone detection"""
        # Test London open kill zone (2 AM - 5 AM NY time)
        london_time = pd.Timestamp('2023-01-02 07:00:00', tz='UTC')  # 2 AM NY
        assert time_features.is_in_killzone(london_time, 'london_open_killzone')
        
        # Test outside kill zone
        outside_time = pd.Timestamp('2023-01-02 15:00:00', tz='UTC')  # 10 AM NY
        assert not time_features.is_in_killzone(outside_time, 'london_open_killzone')
        
        # Test overnight kill zone (Asian)
        asian_time = pd.Timestamp('2023-01-02 00:00:00', tz='UTC')  # 7 PM NY previous day
        assert time_features.is_in_killzone(asian_time, 'asian_killzone')
    
    def test_get_active_killzones(self, time_features):
        """Test getting all active kill zones"""
        # London open time
        london_time = pd.Timestamp('2023-01-02 07:00:00', tz='UTC')  # 2 AM NY
        active_zones = time_features.get_active_killzones(london_time)
        assert 'london_open_killzone' in active_zones
        
        # Overlap time (London close + NY session)
        overlap_time = pd.Timestamp('2023-01-02 15:00:00', tz='UTC')  # 10 AM NY
        active_zones = time_features.get_active_killzones(overlap_time)
        assert 'london_close_killzone' in active_zones
    
    def test_is_in_session(self, time_features):
        """Test session detection"""
        # London session
        london_time = pd.Timestamp('2023-01-02 10:00:00', tz='UTC')  # 5 AM NY
        assert time_features.is_in_session(london_time, 'london')
        
        # NY session
        ny_time = pd.Timestamp('2023-01-02 14:00:00', tz='UTC')  # 9 AM NY
        assert time_features.is_in_session(ny_time, 'new_york')
        
        # Overlap
        overlap_time = pd.Timestamp('2023-01-02 14:00:00', tz='UTC')  # 9 AM NY
        assert time_features.is_in_session(overlap_time, 'overlap')
    
    def test_get_day_of_week_bias(self, time_features):
        """Test day of week bias"""
        # Monday
        monday = pd.Timestamp('2023-01-02 12:00:00', tz='UTC')
        bias = time_features.get_day_of_week_bias(monday)
        assert bias['day'] == 'monday'
        assert bias['day_number'] == 0
        assert bias['pattern'] == 'consolidation_or_false_move'
        assert bias['bias'] == 'range_establishment'
        
        # Thursday
        thursday = pd.Timestamp('2023-01-05 12:00:00', tz='UTC')
        bias = time_features.get_day_of_week_bias(thursday)
        assert bias['day'] == 'thursday'
        assert bias['day_number'] == 3
        assert bias['pattern'] == 'reversal_day'
        assert bias['bias'] == 'reversal_day'
    
    def test_get_time_of_day_features(self, time_features):
        """Test comprehensive time features"""
        timestamp = pd.Timestamp('2023-01-02 13:30:00', tz='UTC')  # 8:30 AM NY
        features = time_features.get_time_of_day_features(timestamp)
        
        assert features['hour'] == 8
        assert features['minute'] == 30
        assert features['is_30min_mark']
        assert 'new_york_open_killzone' in features['active_killzones']
        assert 'new_york' in features['active_sessions']
        assert features['is_new_york_open']
        assert features['power_of_3'] == 'manipulation'
    
    def test_power_of_3_phase(self, time_features):
        """Test Power of 3 phase detection"""
        # Accumulation (early morning)
        early_time = time(2, 0)
        assert time_features._get_power_of_3_phase(early_time) == 'accumulation'
        
        # Manipulation (NY open)
        ny_open_time = time(8, 0)
        assert time_features._get_power_of_3_phase(ny_open_time) == 'manipulation'
        
        # Distribution (NY session)
        ny_session_time = time(14, 0)
        assert time_features._get_power_of_3_phase(ny_session_time) == 'distribution'
    
    def test_is_optimal_trading_time(self, time_features):
        """Test optimal trading time detection"""
        # Optimal time: Tuesday London open
        optimal_time = pd.Timestamp('2023-01-03 07:00:00', tz='UTC')  # 2 AM NY Tuesday
        optimal = time_features.is_optimal_trading_time(optimal_time)
        assert optimal['is_kill_zone']
        assert optimal['is_major_session']
        assert optimal['is_optimal_day']
        assert optimal['overall_optimal']
        
        # Non-optimal: Weekend
        weekend_time = pd.Timestamp('2023-01-07 14:00:00', tz='UTC')  # Saturday
        optimal = time_features.is_optimal_trading_time(weekend_time)
        assert not optimal['is_optimal_day']
        assert not optimal['overall_optimal']
    
    def test_create_time_features(self, time_features, sample_data):
        """Test adding time features to DataFrame"""
        df_with_features = time_features.create_time_features(sample_data)
        
        # Check that features were added
        assert 'hour' in df_with_features.columns
        assert 'day_of_week' in df_with_features.columns
        assert 'in_london_open_killzone' in df_with_features.columns
        assert 'in_london_session' in df_with_features.columns
        assert 'near_midnight_open' in df_with_features.columns
        assert 'power_of_3_phase' in df_with_features.columns
        assert 'is_optimal_time' in df_with_features.columns
        
        # Check some values
        monday_9am = df_with_features.loc['2023-01-02 14:00:00+00:00']  # 9 AM NY
        assert monday_9am['hour'] == 14  # UTC hour
        assert monday_9am['day_of_week'] == 0  # Monday
        assert monday_9am['in_new_york_session']
        assert monday_9am['power_of_3_phase'] == 'manipulation'