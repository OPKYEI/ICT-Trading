# tests/test_features/test_pd_arrays.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.market_structure import MarketStructureAnalyzer, SwingPoint
from src.features.pd_arrays import (
    PriceDeliveryArrays, OrderBlock, FairValueGap, BreakerBlock
)
from src.features.market_structure import MarketStructureAnalyzer, SwingPoint

class TestPriceDeliveryArrays:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with clear patterns"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        
        # Create data with specific patterns for testing
        data = []
        for i in range(len(dates)):
            if i < 20:  # Initial uptrend
                base_price = 1.0 + i * 0.01
                data.append({
                    'open': base_price,
                    'high': base_price + 0.005,
                    'low': base_price - 0.002,
                    'close': base_price + 0.003,
                    'volume': 1000
                })
            elif i < 40:  # Reversal with order blocks
                base_price = 1.2 - (i - 20) * 0.01
                close = base_price - 0.003 if i % 3 == 0 else base_price + 0.003
                data.append({
                    'open': base_price,
                    'high': base_price + 0.005,
                    'low': base_price - 0.005,
                    'close': close,
                    'volume': 1000
                })
            else:  # FVG creation
                base_price = 1.0 + (i - 40) * 0.005
                if i % 5 == 0:  # Create FVG
                    data.append({
                        'open': base_price,
                        'high': base_price + 0.02,  # Large bullish candle
                        'low': base_price,
                        'close': base_price + 0.018,
                        'volume': 2000
                    })
                else:
                    data.append({
                        'open': base_price,
                        'high': base_price + 0.003,
                        'low': base_price - 0.002,
                        'close': base_price + 0.001,
                        'volume': 1000
                    })
        
        df = pd.DataFrame(data, index=dates[:len(data)])
        return df
    
    @pytest.fixture
    def swing_points(self, sample_data):
        """Generate swing points for testing"""
        analyzer = MarketStructureAnalyzer(swing_threshold=3)
        return analyzer.identify_swing_points(sample_data)
    
    @pytest.fixture
    def pd_arrays(self):
        """Create PriceDeliveryArrays instance"""
        return PriceDeliveryArrays(min_order_block_strength=0.001)
    
    def test_identify_order_blocks(self, pd_arrays, sample_data, swing_points):
        """Test order block identification"""
        order_blocks = pd_arrays.identify_order_blocks(sample_data, swing_points)
        
        assert len(order_blocks) > 0
        
        # Check order block properties
        for ob in order_blocks:
            assert ob.type in ['bullish', 'bearish']
            assert ob.high > ob.low
            assert ob.mitigation_level == (ob.open + ob.close) / 2
            
            # Verify it's at a swing point
            candle = sample_data.iloc[ob.start_idx]
            assert candle['high'] == ob.high
            assert candle['low'] == ob.low
    
    def test_identify_fvg(self, pd_arrays, sample_data):
        """Test Fair Value Gap identification"""
        fvgs = pd_arrays.identify_fvg(sample_data)
        
        # Check if any FVGs were found
        if len(fvgs) > 0:
            fvg = fvgs[0]
            assert fvg.type in ['bullish', 'bearish']
            assert fvg.high > fvg.low
            
            # Verify FVG structure
            if fvg.type == 'bullish':
                candle1 = sample_data.iloc[fvg.start_idx]
                candle3 = sample_data.iloc[fvg.end_idx]
                assert candle3['low'] > candle1['high']
            else:
                candle1 = sample_data.iloc[fvg.start_idx]
                candle3 = sample_data.iloc[fvg.end_idx]
                assert candle3['high'] < candle1['low']
    
    def test_identify_breaker_blocks(self, pd_arrays, sample_data, swing_points):
        """Test breaker block identification"""
        order_blocks = pd_arrays.identify_order_blocks(sample_data, swing_points)
        breaker_blocks = pd_arrays.identify_breaker_blocks(
            sample_data, order_blocks, swing_points
        )
        
        # Breaker blocks form when structure breaks
        if len(breaker_blocks) > 0:
            breaker = breaker_blocks[0]
            assert breaker.type in ['bullish', 'bearish']
            assert breaker.origin_type in ['support', 'resistance']
            assert breaker.broken_idx > breaker.start_idx
    
    def test_identify_mitigation_blocks(self, pd_arrays, sample_data, swing_points):
        """Test mitigation block identification"""
        order_blocks = pd_arrays.identify_order_blocks(sample_data, swing_points)
        mitigation_blocks = pd_arrays.identify_mitigation_blocks(sample_data, order_blocks)
        
        # Check mitigation block properties
        for mb in mitigation_blocks:
            assert mb['type'] in ['bullish', 'bearish']
            assert mb['mitigation_idx'] > mb['order_block_idx']
            assert 'mitigation_price' in mb
            assert 'original_ob' in mb
    
    def test_identify_rejection_blocks(self, pd_arrays, sample_data, swing_points):
        """Test rejection block identification"""
        rejection_blocks = pd_arrays.identify_rejection_blocks(sample_data, swing_points)
        
        # Check rejection block properties
        for rb in rejection_blocks:
            assert rb['type'] in ['bullish', 'bearish']
            assert rb['wick_size'] > 0
            assert rb['body_high'] >= rb['body_low']
            
            # Verify wick is significant
            body_size = rb['body_high'] - rb['body_low']
            assert rb['wick_size'] > body_size * 1.5
    
   
    def test_order_block_validation(self, pd_arrays, sample_data):
        """Test that order blocks meet ICT criteria"""
        # Create specific test data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
        
        # Create a clear bullish order block scenario
        # We need a down-closed candle before an up move
        test_data = pd.DataFrame({
            'open': [1.0, 1.005, 1.01, 1.005, 1.002, 1.0, 1.002, 1.01, 1.02, 1.03],
            'high': [1.005, 1.01, 1.015, 1.01, 1.005, 1.002, 1.01, 1.02, 1.03, 1.035],
            'low': [0.995, 1.0, 1.005, 1.0, 0.998, 0.995, 1.0, 1.005, 1.015, 1.025],
            'close': [1.002, 1.008, 1.007, 1.002, 0.999, 1.001, 1.008, 1.018, 1.028, 1.033],
            'volume': [1000] * 10
        }, index=dates)
        
        # Create swing points manually for this test
        swing_points = [
            SwingPoint(index=dates[0], price=0.995, type='low'),
            SwingPoint(index=dates[4], price=0.998, type='low'),
            SwingPoint(index=dates[9], price=1.035, type='high')
        ]
        
        order_blocks = pd_arrays.identify_order_blocks(test_data, swing_points)
        
        # Should find at least one bullish order block
        bullish_obs = [ob for ob in order_blocks if ob.type == 'bullish']
        assert len(bullish_obs) > 0
        
        # Verify the order block properties
        if bullish_obs:
            ob = bullish_obs[0]
            candle = test_data.iloc[ob.start_idx]
            assert candle['close'] < candle['open']  # Should be down-closed candle