# tests/test_data_processing/test_data_loader.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing.data_loader import DataLoader

class TestDataLoader:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        data = {
            'open': np.random.uniform(1.0, 1.1, len(dates)),
            'high': np.random.uniform(1.1, 1.2, len(dates)),
            'low': np.random.uniform(0.9, 1.0, len(dates)),
            'close': np.random.uniform(1.0, 1.1, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }
        df = pd.DataFrame(data, index=dates)
        
        # Ensure high is highest and low is lowest
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    @pytest.fixture
    def loader(self, tmp_path):
        """Create DataLoader instance with temporary directory"""
        return DataLoader(data_path=tmp_path)
    
    def test_validate_data_valid(self, loader, sample_data):
        """Test data validation with valid data"""
        assert loader.validate_data(sample_data) == True
    
    def test_validate_data_missing_values(self, loader, sample_data):
        """Test data validation with missing values"""
        sample_data.loc[sample_data.index[0], 'close'] = np.nan
        assert loader.validate_data(sample_data) == False
    
    def test_validate_data_invalid_prices(self, loader, sample_data):
        """Test data validation with invalid prices"""
        sample_data.loc[sample_data.index[0], 'low'] = -1.0
        assert loader.validate_data(sample_data) == False
    
    def test_validate_data_invalid_high_low(self, loader, sample_data):
        """Test data validation with high < low"""
        sample_data.loc[sample_data.index[0], 'high'] = 0.5
        sample_data.loc[sample_data.index[0], 'low'] = 1.5
        assert loader.validate_data(sample_data) == False
    
    def test_resample_data(self, loader, sample_data):
        """Test data resampling"""
        resampled = loader.resample_data(sample_data, '4h')
        
        # Check that we have fewer rows
        assert len(resampled) < len(sample_data)
        
        # Check that high is max of period
        original_high = sample_data['high'].resample('4h').max()
        pd.testing.assert_series_equal(resampled['high'], original_high.dropna())
        
        # Check that low is min of period
        original_low = sample_data['low'].resample('4h').min()
        pd.testing.assert_series_equal(resampled['low'], original_low.dropna())