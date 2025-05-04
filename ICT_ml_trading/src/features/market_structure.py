# src/features/market_structure.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SwingPoint:
    """Represents a swing high or swing low"""
    index: pd.Timestamp
    price: float
    type: str  # 'high' or 'low'
    
@dataclass
class MarketStructure:
    """Represents the current market structure state"""
    trend: str  # 'bullish', 'bearish', or 'ranging'
    last_hh: Optional[SwingPoint] = None  # Higher High
    last_lh: Optional[SwingPoint] = None  # Lower High  
    last_hl: Optional[SwingPoint] = None  # Higher Low
    last_ll: Optional[SwingPoint] = None  # Lower Low

class MarketStructureAnalyzer:
    """
    Analyzes market structure following ICT concepts:
    - Higher Highs (HH) and Higher Lows (HL) for bullish structure
    - Lower Highs (LH) and Lower Lows (LL) for bearish structure
    - Market Structure Shifts (MSS)
    - Break of Structure (BOS)
    """
    
    def __init__(self, swing_threshold: int = 5):
        """
        Args:
            swing_threshold: Minimum bars on each side to confirm a swing point
        """
        self.swing_threshold = swing_threshold
    
    def identify_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Identify swing highs and lows using the swing threshold
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of SwingPoint objects
        """
        swing_points = []
        
        # Work with the high and low columns
        highs = df['high'].values
        lows = df['low'].values
        
        # Find swing highs
        for i in range(self.swing_threshold, len(df) - self.swing_threshold):
            # Check if current high is highest in the window
            left_window = highs[i-self.swing_threshold:i]
            right_window = highs[i+1:i+self.swing_threshold+1]
            
            if highs[i] > max(left_window) and highs[i] > max(right_window):
                swing_points.append(SwingPoint(
                    index=df.index[i],
                    price=highs[i],
                    type='high'
                ))
        
        # Find swing lows
        for i in range(self.swing_threshold, len(df) - self.swing_threshold):
            # Check if current low is lowest in the window
            left_window = lows[i-self.swing_threshold:i]
            right_window = lows[i+1:i+self.swing_threshold+1]
            
            if lows[i] < min(left_window) and lows[i] < min(right_window):
                swing_points.append(SwingPoint(
                    index=df.index[i],
                    price=lows[i],
                    type='low'
                ))
        
        # Sort by timestamp
        swing_points.sort(key=lambda x: x.index)
        
        return swing_points
    
    def classify_market_structure(self, swing_points: List[SwingPoint]) -> MarketStructure:
        """
        Classify the current market structure based on swing points
        
        Args:
            swing_points: List of identified swing points
            
        Returns:
            MarketStructure object with current state
        """
        if len(swing_points) < 4:
            return MarketStructure(trend='ranging')
        
        # Get the most recent swing points of each type
        recent_highs = [sp for sp in swing_points if sp.type == 'high'][-2:]
        recent_lows = [sp for sp in swing_points if sp.type == 'low'][-2:]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return MarketStructure(trend='ranging')
        
        # Check for higher highs and higher lows (bullish)
        hh = recent_highs[1].price > recent_highs[0].price
        hl = recent_lows[1].price > recent_lows[0].price
        
        # Check for lower highs and lower lows (bearish)
        lh = recent_highs[1].price < recent_highs[0].price
        ll = recent_lows[1].price < recent_lows[0].price
        
        structure = MarketStructure(trend='ranging')
        
        if hh and hl:
            structure.trend = 'bullish'
            structure.last_hh = recent_highs[1]
            structure.last_hl = recent_lows[1]
        elif lh and ll:
            structure.trend = 'bearish'
            structure.last_lh = recent_highs[1]
            structure.last_ll = recent_lows[1]
        else:
            structure.trend = 'ranging'
            # Store the most recent points for reference
            if hh:
                structure.last_hh = recent_highs[1]
            if lh:
                structure.last_lh = recent_highs[1]
            if hl:
                structure.last_hl = recent_lows[1]
            if ll:
                structure.last_ll = recent_lows[1]
        
        return structure
    
    def detect_market_structure_shift(self, 
                                df: pd.DataFrame, 
                                swing_points: List[SwingPoint],
                                current_structure: MarketStructure) -> Dict[str, any]:
        """
        Detect Market Structure Shifts (MSS) and Break of Structure (BOS)
        
        Args:
            df: DataFrame with OHLCV data
            swing_points: List of swing points
            current_structure: Current market structure
            
        Returns:
            Dictionary with MSS/BOS information
        """
        result = {
            'mss_detected': False,
            'bos_detected': False,
            'shift_index': None,
            'shift_price': None,
            'shift_type': None  # 'bullish_to_bearish' or 'bearish_to_bullish'
        }
        
        if len(swing_points) < 4:
            return result
        
        # Get the latest price action
        latest_price = df['close'].iloc[-1]
        latest_idx = df.index[-1]
        
        # Check for MSS based on current structure
        if current_structure.trend == 'bullish':
            # Look for break below last HL for bearish MSS
            if current_structure.last_hl:
                # Check if price has broken below last HL
                for i in range(len(df) - 1, -1, -1):
                    if df.iloc[i]['low'] < current_structure.last_hl.price:
                        result['mss_detected'] = True
                        result['shift_index'] = df.index[i]
                        result['shift_price'] = current_structure.last_hl.price
                        result['shift_type'] = 'bullish_to_bearish'
                        break
                        
        elif current_structure.trend == 'bearish':
            # Look for break above last LH for bullish MSS
            if current_structure.last_lh:
                # Check if price has broken above last LH
                for i in range(len(df) - 1, -1, -1):
                    if df.iloc[i]['high'] > current_structure.last_lh.price:
                        result['mss_detected'] = True
                        result['shift_index'] = df.index[i]
                        result['shift_price'] = current_structure.last_lh.price
                        result['shift_type'] = 'bearish_to_bullish'
                        break
        
        # Check for BOS (continuation)
        if current_structure.trend == 'bullish' and current_structure.last_hh:
            for i in range(len(df) - 1, -1, -1):
                if df.iloc[i]['high'] > current_structure.last_hh.price:
                    result['bos_detected'] = True
                    result['shift_index'] = df.index[i]
                    result['shift_price'] = current_structure.last_hh.price
                    break
                    
        elif current_structure.trend == 'bearish' and current_structure.last_ll:
            for i in range(len(df) - 1, -1, -1):
                if df.iloc[i]['low'] < current_structure.last_ll.price:
                    result['bos_detected'] = True
                    result['shift_index'] = df.index[i]
                    result['shift_price'] = current_structure.last_ll.price
                    break
        
        return result
    
    def identify_ranges(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> List[Dict]:
        """
        Identify consolidation ranges in the market
        
        Args:
            df: DataFrame with OHLCV data
            swing_points: List of swing points
            
        Returns:
            List of range dictionaries with start, end, high, and low
        """
        ranges = []
        
        if len(swing_points) < 4:
            return ranges
        
        # Parameters for range detection
        min_touches = 2  # Minimum touches of support/resistance
        tolerance = 0.001  # 0.1% tolerance for price levels
        
        # Group nearby swing points to find potential ranges
        for i in range(len(swing_points) - 3):
            potential_range_points = swing_points[i:i+4]
            
            highs = [p for p in potential_range_points if p.type == 'high']
            lows = [p for p in potential_range_points if p.type == 'low']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check if highs are within tolerance
                high_prices = [h.price for h in highs]
                high_avg = np.mean(high_prices)
                high_deviation = np.std(high_prices) / high_avg
                
                # Check if lows are within tolerance
                low_prices = [l.price for l in lows]
                low_avg = np.mean(low_prices)
                low_deviation = np.std(low_prices) / low_avg
                
                if high_deviation < tolerance and low_deviation < tolerance:
                    # We have a potential range
                    range_start = potential_range_points[0].index
                    range_end = potential_range_points[-1].index
                    
                    # Extend range to current if price is still within
                    current_price = df['close'].iloc[-1]
                    if low_avg <= current_price <= high_avg:
                        range_end = df.index[-1]
                    
                    ranges.append({
                        'start': range_start,
                        'end': range_end,
                        'high': high_avg,
                        'low': low_avg,
                        'midpoint': (high_avg + low_avg) / 2
                    })
        
        return ranges