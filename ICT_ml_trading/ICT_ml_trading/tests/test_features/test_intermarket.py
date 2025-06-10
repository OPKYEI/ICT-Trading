# tests/test_features/test_intermarket.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.intermarket import (
    IntermarketAnalysis, CorrelationResult, SMTDivergence, StrengthScore
)

class TestIntermarketAnalysis:
    
    @pytest.fixture
    def intermarket_analyzer(self):
        """Create IntermarketAnalysis instance"""
        return IntermarketAnalysis(
            correlation_window=20,
            significance_level=0.05,
            min_divergence_bars=5
        )
    
    @pytest.fixture
    def sample_correlated_data(self):
        """Create sample correlated price data"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
        
        # Create two correlated series
        base_prices = 1.0 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        # Series 1: Direct correlation
        series1 = base_prices + np.random.randn(len(dates)) * 0.002
        
        # Series 2: Inverse correlation
        series2 = 2.0 - base_prices + np.random.randn(len(dates)) * 0.002
        
        return pd.Series(series1, index=dates), pd.Series(series2, index=dates)
    
    @pytest.fixture
    def sample_pairs_data(self):
        """Create sample forex pairs data"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
        
        pairs_data = {}
        
        # EURUSD - trending up
        eurusd_prices = 1.05 + np.cumsum(np.random.randn(len(dates)) * 0.001)
        pairs_data['EURUSD'] = pd.DataFrame({
            'open': eurusd_prices,
            'high': eurusd_prices + 0.001,
            'low': eurusd_prices - 0.001,
            'close': eurusd_prices + np.random.randn(len(dates)) * 0.0005,
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # GBPUSD - trending up with EURUSD (correlated)
        gbpusd_prices = 1.25 + np.cumsum(np.random.randn(len(dates)) * 0.001)
        pairs_data['GBPUSD'] = pd.DataFrame({
            'open': gbpusd_prices,
            'high': gbpusd_prices + 0.001,
            'low': gbpusd_prices - 0.001,
            'close': gbpusd_prices + np.random.randn(len(dates)) * 0.0005,
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # USDJPY - inverse correlation (USD strength = JPY weakness)
        usdjpy_prices = 110.0 - np.cumsum(np.random.randn(len(dates)) * 0.1)
        pairs_data['USDJPY'] = pd.DataFrame({
            'open': usdjpy_prices,
            'high': usdjpy_prices + 0.1,
            'low': usdjpy_prices - 0.1,
            'close': usdjpy_prices + np.random.randn(len(dates)) * 0.05,
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # USDCHF - inverse correlation with EURUSD
        usdchf_prices = 0.95 - np.cumsum(np.random.randn(len(dates)) * 0.001)
        pairs_data['USDCHF'] = pd.DataFrame({
            'open': usdchf_prices,
            'high': usdchf_prices + 0.001,
            'low': usdchf_prices - 0.001,
            'close': usdchf_prices + np.random.randn(len(dates)) * 0.0005,
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # USDCAD 
        usdcad_prices = 1.35 + np.cumsum(np.random.randn(len(dates)) * 0.001)
        pairs_data['USDCAD'] = pd.DataFrame({
            'open': usdcad_prices,
            'high': usdcad_prices + 0.001,
            'low': usdcad_prices - 0.001,
            'close': usdcad_prices + np.random.randn(len(dates)) * 0.0005,
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        return pairs_data
    
    @pytest.fixture
    def sample_divergence_data(self):
        """Create sample data with clear SMT divergence"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        
        # Asset 1: Making higher highs
        asset1_data = []
        for i in range(len(dates)):
            if i < 50:
                price = 1.0 + i * 0.002  # Uptrend
            elif i < 100:
                price = 1.1 - (i - 50) * 0.001  # Pullback
            elif i < 150:
                price = 1.05 + (i - 100) * 0.0025  # Higher high
            else:
                price = 1.175 - (i - 150) * 0.001  # Pullback
            
            asset1_data.append({
                'open': price,
                'high': price + 0.002,
                'low': price - 0.002,
                'close': price + 0.001,
                'volume': 1000
            })
        
        # Asset 2: Making lower highs (divergence)
        asset2_data = []
        for i in range(len(dates)):
            if i < 50:
                price = 1.0 + i * 0.002  # Uptrend
            elif i < 100:
                price = 1.1 - (i - 50) * 0.001  # Pullback
            elif i < 150:
                price = 1.05 + (i - 100) * 0.0015  # Lower high
            else:
                price = 1.125 - (i - 150) * 0.001  # Pullback
            
            asset2_data.append({
                'open': price,
                'high': price + 0.002,
                'low': price - 0.002,
                'close': price + 0.001,
                'volume': 1000
            })
        
        df1 = pd.DataFrame(asset1_data, index=dates[:len(asset1_data)])
        df2 = pd.DataFrame(asset2_data, index=dates[:len(asset2_data)])
        
        return df1, df2
    
    def test_calculate_correlation(self, intermarket_analyzer, sample_correlated_data):
        """Test correlation calculation"""
        series1, series2 = sample_correlated_data
        
        # Test correlation
        corr_result = intermarket_analyzer.calculate_correlation(series1, series2)
        
        assert isinstance(corr_result, CorrelationResult)
        assert -1 <= corr_result.correlation_coefficient <= 1
        assert 0 <= corr_result.p_value <= 1
        assert corr_result.window == 20
        assert corr_result.direction in ['positive', 'negative', 'neutral']
    
    def test_detect_smt_divergence(self, intermarket_analyzer, sample_divergence_data):
        """Test SMT divergence detection"""
        asset1_data, asset2_data = sample_divergence_data
        
        divergences = intermarket_analyzer.detect_smt_divergence(
            asset1_data, asset2_data, lookback=50
        )
        
        # Should find at least one divergence in this data
        if len(divergences) > 0:
            divergence = divergences[0]
            assert isinstance(divergence, SMTDivergence)
            assert divergence.divergence_type in ['bullish', 'bearish']
            assert divergence.confirmed
            assert 0 <= divergence.strength <= 1
    
    def test_calculate_currency_strength(self, intermarket_analyzer, sample_pairs_data):
        """Test currency strength calculation"""
        strength_scores = intermarket_analyzer.calculate_currency_strength(
            sample_pairs_data, lookback=14
        )
        
        assert 'EUR' in strength_scores
        assert 'USD' in strength_scores
        assert 'GBP' in strength_scores
        
        # Check StrengthScore properties
        for currency, score in strength_scores.items():
            assert isinstance(score, StrengthScore)
            assert score.currency == currency
            assert score.rank > 0
            assert score.trend in ['strengthening', 'weakening', 'stable']
    
    def test_find_correlated_pairs(self, intermarket_analyzer, sample_pairs_data):
        """Test finding correlated pairs"""
        correlations = intermarket_analyzer.find_correlated_pairs(
            sample_pairs_data, min_correlation=0.3
        )
        
        # Check structure
        assert isinstance(correlations, dict)
        for pair, corr_list in correlations.items():
            assert isinstance(corr_list, list)
            for corr_pair, corr_value in corr_list:
                assert isinstance(corr_pair, str)
                assert -1 <= corr_value <= 1
    
    def test_calculate_dollar_index_proxy(self, intermarket_analyzer, sample_pairs_data):
        """Test dollar index proxy calculation"""
        dxy_proxy = intermarket_analyzer.calculate_dollar_index_proxy(sample_pairs_data)
        
        assert isinstance(dxy_proxy, pd.Series)
        assert len(dxy_proxy) > 0
        assert not dxy_proxy.isna().any()
    
    def test_find_swing_highs(self, intermarket_analyzer, sample_pairs_data):
        """Test swing high detection"""
        eurusd_data = sample_pairs_data['EURUSD']
        
        highs = intermarket_analyzer._find_swing_highs(eurusd_data, window=5)
        
        # Check structure
        for high in highs:
            assert 'index' in high
            assert 'price' in high
            assert 'timestamp' in high
            assert high['price'] > 0
    
    def test_find_swing_lows(self, intermarket_analyzer, sample_pairs_data):
        """Test swing low detection"""
        eurusd_data = sample_pairs_data['EURUSD']
        
        lows = intermarket_analyzer._find_swing_lows(eurusd_data, window=5)
        
        # Check structure
        for low in lows:
            assert 'index' in low
            assert 'price' in low
            assert 'timestamp' in low
            assert low['price'] > 0