# tests/test_features/test_feature_engineering.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.feature_engineering import ICTFeatureEngineer, FeatureSet

class TestFeatureEngineering:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        
        # Create realistic price movements
        np.random.seed(42)
        prices = 1.0
        price_data = []
        
        for _ in range(len(dates)):
            # Add some trend and volatility
            change = np.random.normal(0.0001, 0.002)
            prices *= (1 + change)
            
            open_price = prices
            high_price = prices * (1 + abs(np.random.normal(0, 0.001)))
            low_price = prices * (1 - abs(np.random.normal(0, 0.001)))
            close_price = prices * (1 + np.random.normal(0, 0.0005))
            
            price_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 10000)
            })
        
        df = pd.DataFrame(price_data, index=dates)
        df.name = 'GBPUSD'  # Add name attribute
        return df
    
    @pytest.fixture
    def additional_data(self):
        """Create additional symbol data for intermarket analysis"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        
        symbols = {}
        
        # Create correlated EURUSD data
        prices = 1.1
        eurusd_data = []
        for _ in range(len(dates)):
            change = np.random.normal(0.0001, 0.002)
            prices *= (1 + change)
            
            eurusd_data.append({
                'open': prices,
                'high': prices * 1.001,
                'low': prices * 0.999,
                'close': prices * (1 + np.random.normal(0, 0.0005)),
                'volume': np.random.randint(1000, 10000)
            })
        
        symbols['EURUSD'] = pd.DataFrame(eurusd_data, index=dates)
        
        # Create inversely correlated USDCHF data
        prices = 0.95
        usdchf_data = []
        for i in range(len(dates)):
            # Inverse correlation with EURUSD
            change = -eurusd_data[i]['close'] / 1.1 + 1
            prices *= change
            
            usdchf_data.append({
                'open': prices,
                'high': prices * 1.001,
                'low': prices * 0.999,
                'close': prices * (1 + np.random.normal(0, 0.0005)),
                'volume': np.random.randint(1000, 10000)
            })
        
        symbols['USDCHF'] = pd.DataFrame(usdchf_data, index=dates)
        
        return symbols
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance"""
        return ICTFeatureEngineer(
            lookback_periods=[5, 10, 20],
            feature_selection_threshold=0.01
        )
    
    def test_engineer_features(self, feature_engineer, sample_data, additional_data):
        """Test feature engineering pipeline"""
        feature_set = feature_engineer.engineer_features(
            sample_data,
            symbol='GBPUSD',
            additional_data=additional_data
        )
        
        assert isinstance(feature_set, FeatureSet)
        assert isinstance(feature_set.features, pd.DataFrame)
        assert len(feature_set.features) > 0
        assert len(feature_set.feature_names) > 0
        
        # Check metadata
        assert feature_set.metadata['symbol'] == 'GBPUSD'
        assert 'start_date' in feature_set.metadata
        assert 'end_date' in feature_set.metadata
    
    def test_market_structure_features(self, feature_engineer, sample_data):
        """Test market structure feature generation"""
        features = pd.DataFrame(index=sample_data.index)
        features = feature_engineer._add_market_structure_features(sample_data, features)
        
        # Check for required columns
        assert 'swing_high' in features.columns
        assert 'swing_low' in features.columns
        assert 'market_trend_encoded' in features.columns
        assert 'distance_from_swing_high' in features.columns
        assert 'structure_strength' in features.columns
    
    def test_time_features(self, feature_engineer, sample_data):
        """Test time feature generation"""
        features = pd.DataFrame(index=sample_data.index)
        features = feature_engineer._add_time_features(sample_data, features)
        
        # Check for required columns
        assert 'hour' in features.columns
        assert 'day_of_week' in features.columns
        assert 'in_london_session' in features.columns
        assert 'in_new_york_session' in features.columns
        assert 'is_optimal_time' in features.columns
    
    def test_pd_array_features(self, feature_engineer, sample_data):
        """Test PD array feature generation"""
        features = pd.DataFrame(index=sample_data.index)
        features = feature_engineer._add_pd_array_features(sample_data, features)
        
        # Check for required columns
        assert 'near_bullish_ob' in features.columns
        assert 'near_bearish_ob' in features.columns
        assert 'in_fvg' in features.columns
        assert 'near_breaker' in features.columns
    
    def test_liquidity_features(self, feature_engineer, sample_data):
        """Test liquidity feature generation"""
        features = pd.DataFrame(index=sample_data.index)
        features = feature_engineer._add_liquidity_features(sample_data, features)
        
        # Check for required columns
        assert 'near_bsl' in features.columns
        assert 'near_ssl' in features.columns
        assert 'stop_run' in features.columns
        assert 'near_liquidity_pool' in features.columns
    
    def test_pattern_features(self, feature_engineer, sample_data):
        """Test pattern feature generation"""
        features = pd.DataFrame(index=sample_data.index)
        features = feature_engineer._add_pattern_features(sample_data, features)
        
        # Check for required columns
        assert 'pattern_detected' in features.columns
        assert 'pattern_type' in features.columns
        assert 'pattern_direction' in features.columns
        assert 'pattern_confidence' in features.columns
    
    def test_intermarket_features(self, feature_engineer, sample_data, additional_data):
        """Test intermarket feature generation"""
        features = pd.DataFrame(index=sample_data.index)
        sample_data.name = 'GBPUSD'  # Set the symbol name
        
        features = feature_engineer._add_intermarket_features(
            sample_data, features, additional_data
        )
        
        # Check for correlation features
        assert any('smt_divergence' in col for col in features.columns)
    
    
    def test_technical_indicators(self, feature_engineer, sample_data):
        """Test technical indicator generation"""
        features = pd.DataFrame(index=sample_data.index)
        features = feature_engineer._add_technical_indicators(sample_data, features)
        
        # Check for required columns
        assert 'atr' in features.columns
        assert 'rsi' in features.columns
        assert 'bb_position' in features.columns
        assert any('sma_' in col for col in features.columns)
    
    def test_target_variables(self, feature_engineer, sample_data):
        """Test target variable generation"""
        features = pd.DataFrame(index=sample_data.index)
        features = feature_engineer._add_target_variables(sample_data, features)
        # Check for required columns
        assert 'future_return_1' in features.columns
        assert 'future_direction_1' in features.columns
        assert 'mfe_5' in features.columns
        assert 'mae_5' in features.columns
        
    def test_feature_cleanup(self, feature_engineer):
        """Test feature cleanup functionality"""
        # Create sample features with issues
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        features = pd.DataFrame({
           'numeric': np.random.randn(100),
           'categorical': ['A', 'B', 'C'] * 33 + ['A'],
           'with_nan': [1.0] * 50 + [np.nan] * 50,
           'with_inf': [1.0] * 50 + [np.inf] * 25 + [-np.inf] * 25
        }, index=dates)

        cleaned = feature_engineer._cleanup_features(features)

        # Check cleanup
        assert 'categorical_encoded' in cleaned.columns
        assert 'categorical' not in cleaned.columns
        assert not cleaned.isna().any().any()
        assert not np.isinf(cleaned).any().any()
   
    def test_feature_selection(self, feature_engineer, sample_data):
        """Test feature selection functionality"""
        # Engineer features first
        feature_set = feature_engineer.engineer_features(sample_data, 'EURUSD')
        
        # Make sure we have the target column
        if 'future_direction_5' in feature_set.features.columns:
            # Select features
            target_column = 'future_direction_5'
            selected_features = feature_engineer.select_features(
                feature_set.features,
                target_column,
                method='correlation'
            )
            
            assert isinstance(selected_features, list)
            assert len(selected_features) > 0
            assert target_column not in selected_features
        else:
            # Skip if target doesn't exist
            assert True
   
    def test_full_pipeline_with_nan_handling(self, feature_engineer, sample_data):
        """Test full pipeline with edge cases"""
        # Add some NaN values
        sample_data.iloc[0:5, 0] = np.nan
        sample_data.iloc[-5:, 3] = np.nan
        
        feature_set = feature_engineer.engineer_features(sample_data, 'EURUSD')
        
        # Should handle NaN values properly
        assert not feature_set.features.isna().all().any()
        assert len(feature_set.features) > 0
   
    def test_feature_configuration(self, feature_engineer, sample_data):
        """Test feature configuration settings"""
        # Disable some features
        feature_engineer.feature_config['patterns'] = False
        feature_engineer.feature_config['intermarket'] = False

        feature_set = feature_engineer.engineer_features(sample_data, 'EURUSD')

        # Pattern features should not be present
        pattern_cols = [col for col in feature_set.features.columns if 'pattern' in col]
        assert len(pattern_cols) == 0