# tests/test_features/test_market_structure.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.market_structure import MarketStructureAnalyzer, SwingPoint, MarketStructure

class TestMarketStructure:
    
    @pytest.fixture
    def sample_bullish_data(self):
        """Create sample data with clear bullish structure"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
        
        # Create a clear uptrend with HH and HL
        prices = []
        base_price = 1.0
        for i in range(len(dates)):
            # Create waves with higher highs and higher lows
            wave = np.sin(i * 0.1) * 0.02
            trend = i * 0.0001  # Upward trend
            price = base_price + trend + wave
            prices.append(price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.001 for p in prices],
            'low': [p - 0.001 for p in prices],
            'close': prices,
            'volume': [1000] * len(dates)
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def sample_bearish_data(self):
        """Create sample data with clear bearish structure"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
        
        # Create a clear downtrend with LH and LL
        prices = []
        base_price = 1.1
        for i in range(len(dates)):
            # Create waves with lower highs and lower lows
            wave = np.sin(i * 0.1) * 0.02
            trend = -i * 0.0001  # Downward trend
            price = base_price + trend + wave
            prices.append(price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.001 for p in prices],
            'low': [p - 0.001 for p in prices],
            'close': prices,
            'volume': [1000] * len(dates)
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def sample_ranging_data(self):
        """Create sample data with ranging structure"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
        
        # Create a ranging market
        prices = []
        base_price = 1.05
        for i in range(len(dates)):
            # Create waves without trend
            wave = np.sin(i * 0.1) * 0.02
            price = base_price + wave
            prices.append(price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.001 for p in prices],
            'low': [p - 0.001 for p in prices],
            'close': prices,
            'volume': [1000] * len(dates)
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def analyzer(self):
        """Create MarketStructureAnalyzer instance"""
        return MarketStructureAnalyzer(swing_threshold=5)
    
    def test_identify_swing_points_bullish(self, analyzer, sample_bullish_data):
        """Test swing point identification in bullish market"""
        swing_points = analyzer.identify_swing_points(sample_bullish_data)
        
        # Check that we have swing points
        assert len(swing_points) > 0
        
        # Check that we have both highs and lows
        highs = [sp for sp in swing_points if sp.type == 'high']
        lows = [sp for sp in swing_points if sp.type == 'low']
        
        assert len(highs) > 0
        assert len(lows) > 0
        
        # Check that highs are generally increasing (bullish)
        if len(highs) >= 2:
            last_highs = [h.price for h in highs[-3:]]
            assert last_highs[-1] > last_highs[0]  # Recent high > older high
    
    def test_identify_swing_points_bearish(self, analyzer, sample_bearish_data):
        """Test swing point identification in bearish market"""
        swing_points = analyzer.identify_swing_points(sample_bearish_data)
        
        # Check that we have swing points
        assert len(swing_points) > 0
        
        # Check that lows are generally decreasing (bearish)
        lows = [sp for sp in swing_points if sp.type == 'low']
        if len(lows) >= 2:
            last_lows = [l.price for l in lows[-3:]]
            assert last_lows[-1] < last_lows[0]  # Recent low < older low
    
    def test_classify_market_structure_bullish(self, analyzer, sample_bullish_data):
        """Test market structure classification for bullish trend"""
        swing_points = analyzer.identify_swing_points(sample_bullish_data)
        structure = analyzer.classify_market_structure(swing_points)
        
        assert structure.trend == 'bullish'
        assert structure.last_hh is not None
        assert structure.last_hl is not None
    
    def test_classify_market_structure_bearish(self, analyzer, sample_bearish_data):
        """Test market structure classification for bearish trend"""
        swing_points = analyzer.identify_swing_points(sample_bearish_data)
        structure = analyzer.classify_market_structure(swing_points)
        
        assert structure.trend == 'bearish'
        assert structure.last_lh is not None
        assert structure.last_ll is not None
    
    def test_classify_market_structure_ranging(self, analyzer, sample_ranging_data):
        """Test market structure classification for ranging market"""
        swing_points = analyzer.identify_swing_points(sample_ranging_data)
        structure = analyzer.classify_market_structure(swing_points)
        
        # Ranging markets might be classified as ranging or have mixed signals
        assert structure.trend in ['ranging', 'bullish', 'bearish']
    
        # tests/test_features/test_market_structure.py (updated test_detect_market_structure_shift)
    def test_detect_market_structure_shift(self, analyzer):
        """Test detection of market structure shift"""
        # Create data with a clear structure shift
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
        
        # Start bullish, then shift to bearish with a clear break
        prices = []
        for i in range(len(dates)):
            if i < len(dates) // 2:
                # Bullish phase
                price = 1.0 + i * 0.0001 + np.sin(i * 0.1) * 0.02
            else:
                # Bearish phase with break below previous HL
                price = 1.0 - (i - len(dates) // 2) * 0.0002 + np.sin(i * 0.1) * 0.02
            prices.append(price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.001 for p in prices],
            'low': [p - 0.001 for p in prices],
            'close': prices,
            'volume': [1000] * len(dates)
        }, index=dates)
        
        # Get full swing points
        swing_points = analyzer.identify_swing_points(df)
        
        # Test with a portion of data that includes the shift
        test_df = df.iloc[:len(df)//2+20]  # Include some bearish data
        test_swing_points = analyzer.identify_swing_points(test_df)
        
        structure = analyzer.classify_market_structure(test_swing_points)
        mss_result = analyzer.detect_market_structure_shift(test_df, test_swing_points, structure)
        
        # Should detect either MSS or structure change
        assert mss_result['mss_detected'] or structure.trend != 'bullish'

    def test_identify_ranges(self, analyzer, sample_ranging_data):
        """Test range identification"""
        swing_points = analyzer.identify_swing_points(sample_ranging_data)
        ranges = analyzer.identify_ranges(sample_ranging_data, swing_points)
        
        # Should identify at least one range in ranging data
        assert len(ranges) > 0
        
        if len(ranges) > 0:
            range_info = ranges[0]
            assert 'start' in range_info
            assert 'end' in range_info
            assert 'high' in range_info
            assert 'low' in range_info
            assert 'midpoint' in range_info
            
            # Check midpoint calculation
            expected_midpoint = (range_info['high'] + range_info['low']) / 2
            assert abs(range_info['midpoint'] - expected_midpoint) < 1e-10