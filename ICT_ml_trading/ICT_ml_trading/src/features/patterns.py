# src/features/patterns.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TradePattern:
    """Represents a detected trading pattern"""
    pattern_type: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    start_idx: int
    end_idx: int
    direction: str  # 'long' or 'short'
    notes: str = ""

class PatternRecognition:
    """
    Detects ICT trading patterns including:
    - OTE (Optimal Trade Entry)
    - Turtle Soup
    - Breaker Patterns
    - Fair Value Gap Entries
    - Market Structure Shift Entries
    """
    
    def __init__(self, 
                 ote_fib_levels: List[float] = [0.62, 0.705, 0.79],
                 min_swing_size: float = 0.001,
                 atr_multiplier: float = 1.5):
        """
        Args:
            ote_fib_levels: Fibonacci levels for OTE zone
            min_swing_size: Minimum price movement for valid swing
            atr_multiplier: ATR multiplier for stop loss calculation
        """
        self.ote_fib_levels = ote_fib_levels
        self.min_swing_size = min_swing_size
        self.atr_multiplier = atr_multiplier
    
    # src/features/patterns.py
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Handle the first row separately to avoid NaN
        if len(df) < 2:
            # Return a series of zeros if not enough data
            return pd.Series(0, index=df.index)
        
        # True Range calculation
        prev_close = close.shift(1)
        
        # For the first row, use the current row's high-low range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        # Fill NaN values in tr2 and tr3 for the first row
        tr2 = tr2.fillna(tr1)
        tr3 = tr3.fillna(tr1)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR calculation
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def find_ote_patterns(self, df: pd.DataFrame, 
                         swing_points: List,
                         fvgs: List,
                         order_blocks: List) -> List[TradePattern]:
        """
        Find OTE (Optimal Trade Entry) patterns
        
        Criteria:
        1. Price retraces to 62-79% Fibonacci level
        2. Confluence with FVG or Order Block
        3. Market structure supports direction
        """
        patterns = []
        atr = self.calculate_atr(df)
        
        # Look for retracements in recent swings
        for i in range(2, len(swing_points)):
            swing_low = None
            swing_high = None
            
            # Find the last swing low and high
            for j in range(i-1, -1, -1):
                if swing_points[j].type == 'low' and swing_low is None:
                    swing_low = swing_points[j]
                elif swing_points[j].type == 'high' and swing_high is None:
                    swing_high = swing_points[j]
                
                if swing_low and swing_high:
                    break
            
            if not (swing_low and swing_high):
                continue
            
            # Determine trend direction
            if swing_low.index < swing_high.index:  # Uptrend
                impulse_start = swing_low
                impulse_end = swing_high
                direction = 'long'
            else:  # Downtrend
                impulse_start = swing_high
                impulse_end = swing_low
                direction = 'short'
            
            # Calculate Fibonacci levels
            impulse_range = abs(impulse_end.price - impulse_start.price)
            if impulse_range < self.min_swing_size:
                continue
            
            fib_levels = {}
            for level in self.ote_fib_levels:
                if direction == 'long':
                    fib_levels[level] = impulse_end.price - (impulse_range * level)
                else:
                    fib_levels[level] = impulse_end.price + (impulse_range * level)
            
            # Look for price entering OTE zone
            ote_zone_high = max(fib_levels.values())
            ote_zone_low = min(fib_levels.values())
            
            # Check recent price action for OTE entry
            recent_bars = df.iloc[max(0, i-10):i+1]
            
            for idx, bar in recent_bars.iterrows():
                bar_idx = df.index.get_loc(idx)
                
                # Check if price entered OTE zone
                if direction == 'long' and bar['low'] <= ote_zone_high and bar['high'] >= ote_zone_low:
                    # Check for FVG or Order Block confluence
                    confluence_found = False
                    confidence = 0.5
                    
                    # Check FVGs
                    for fvg in fvgs:
                        if fvg.type == 'bullish' and abs(fvg.low - bar['low']) < 0.001:
                            confluence_found = True
                            confidence += 0.2
                            break
                    
                    # Check Order Blocks
                    for ob in order_blocks:
                        if ob.type == 'bullish' and bar['low'] >= ob.low and bar['low'] <= ob.high:
                            confluence_found = True
                            confidence += 0.3
                            break
                    
                    if confluence_found:
                        entry_price = bar['close']
                        stop_loss = ote_zone_low - atr[idx] * self.atr_multiplier
                        tp1 = impulse_end.price
                        tp2 = impulse_end.price + (impulse_range * 0.5)
                        tp3 = impulse_end.price + impulse_range
                        
                        patterns.append(TradePattern(
                            pattern_type='ote',
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=[tp1, tp2, tp3],
                            start_idx=df.index.get_loc(impulse_start.index),
                            end_idx=bar_idx,
                            direction=direction,
                            notes=f"OTE entry at {level:.3f} Fibonacci level"
                        ))
                
                elif direction == 'short' and bar['high'] >= ote_zone_low and bar['low'] <= ote_zone_high:
                    # Similar logic for short trades
                    confluence_found = False
                    confidence = 0.5
                    
                    # Check FVGs
                    for fvg in fvgs:
                        if fvg.type == 'bearish' and abs(fvg.high - bar['high']) < 0.001:
                            confluence_found = True
                            confidence += 0.2
                            break
                    
                    # Check Order Blocks
                    for ob in order_blocks:
                        if ob.type == 'bearish' and bar['high'] <= ob.high and bar['high'] >= ob.low:
                            confluence_found = True
                            confidence += 0.3
                            break
                    
                    if confluence_found:
                        entry_price = bar['close']
                        stop_loss = ote_zone_high + atr[idx] * self.atr_multiplier
                        tp1 = impulse_end.price
                        tp2 = impulse_end.price - (impulse_range * 0.5)
                        tp3 = impulse_end.price - impulse_range
                        
                        patterns.append(TradePattern(
                            pattern_type='ote',
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=[tp1, tp2, tp3],
                            start_idx=df.index.get_loc(impulse_start.index),
                            end_idx=bar_idx,
                            direction=direction,
                            notes=f"OTE entry at {level:.3f} Fibonacci level"
                        ))
        
        return patterns
    
    def find_turtle_soup_patterns(self, df: pd.DataFrame,
                                liquidity_levels: List,
                                swing_points: List) -> List[TradePattern]:
        """
        Find Turtle Soup patterns (false breakouts)

        Criteria:
        1. Price sweeps liquidity (stop hunt)
        2. Quick reversal after sweep
        3. Closes back within range
        """
        patterns = []
        atr = self.calculate_atr(df)

        for i, level in enumerate(liquidity_levels):
            if not level.swept:
                continue

            sweep_idx = df.index.get_loc(level.swept_index)
            if isinstance(sweep_idx, slice):
                sweep_idx = sweep_idx.start

            sweep_bar = df.iloc[sweep_idx]

            for j in range(1, min(6, len(df) - sweep_idx)):
                current_idx = sweep_idx + j
                current_bar = df.iloc[current_idx]

                if level.type == 'BSL':
                    if current_bar['close'] < sweep_bar['close'] - (sweep_bar['high'] - sweep_bar['low']):
                        entry_price = current_bar['close']
                        stop_loss = sweep_bar['high'] + atr[current_idx] * 0.5

                        recent_low = next((sp for sp in reversed(swing_points)
                                           if sp.type == 'low' and sp.index < df.index[current_idx]), None)

                        if recent_low:
                            tp1 = recent_low.price
                            tp2 = recent_low.price - (sweep_bar['high'] - recent_low.price) * 0.5
                            tp3 = recent_low.price - (sweep_bar['high'] - recent_low.price)

                            patterns.append(TradePattern(
                                pattern_type='turtle_soup',
                                confidence=0.8,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=[tp1, tp2, tp3],
                                start_idx=sweep_idx,
                                end_idx=current_idx,
                                direction='short',
                                notes="Buy-side liquidity sweep (Turtle Soup)"
                            ))

                elif level.type == 'SSL':
                    if current_bar['close'] > sweep_bar['close'] + (sweep_bar['high'] - sweep_bar['low']):
                        entry_price = current_bar['close']
                        stop_loss = sweep_bar['low'] - atr[current_idx] * 0.5

                        recent_high = next((sp for sp in reversed(swing_points)
                                            if sp.type == 'high' and sp.index < df.index[current_idx]), None)

                        if recent_high:
                            tp1 = recent_high.price
                            tp2 = recent_high.price + (recent_high.price - sweep_bar['low']) * 0.5
                            tp3 = recent_high.price + (recent_high.price - sweep_bar['low'])

                            patterns.append(TradePattern(
                                pattern_type='turtle_soup',
                                confidence=0.8,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=[tp1, tp2, tp3],
                                start_idx=sweep_idx,
                                end_idx=current_idx,
                                direction='long',
                                notes="Sell-side liquidity sweep (Turtle Soup)"
                            ))

            return patterns

    
    def find_breaker_patterns(self, df: pd.DataFrame,
                            breaker_blocks: List,
                            swing_points: List) -> List[TradePattern]:
        """
        Find Breaker Block patterns
        
        Criteria:
        1. Price breaks structure
        2. Returns to breaker block
        3. Respects breaker as new support/resistance
        """
        patterns = []
        atr = self.calculate_atr(df)
        
        for breaker in breaker_blocks:
            # Look for price returning to breaker
            for i in range(breaker.broken_idx + 1, len(df)):
                bar = df.iloc[i]
                
                if breaker.type == 'bullish':
                    # Price returning to bullish breaker from above
                    if bar['low'] <= breaker.high and bar['low'] >= breaker.low:
                        entry_price = bar['close']
                        stop_loss = breaker.low - atr.iloc[i] * 0.5
                        
                        # Find recent high for targets
                        recent_high = None
                        for sp in reversed(swing_points):
                            if sp.type == 'high' and sp.index > df.index[breaker.broken_idx]:
                                recent_high = sp
                                break
                        
                        if recent_high:
                            tp_range = recent_high.price - breaker.high
                            tp1 = entry_price + tp_range * 0.5
                            tp2 = entry_price + tp_range
                            tp3 = entry_price + tp_range * 1.5
                            
                            patterns.append(TradePattern(
                                pattern_type='breaker',
                                confidence=0.75,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=[tp1, tp2, tp3],
                                start_idx=breaker.start_idx,
                                end_idx=i,
                                direction='long',
                                notes="Bullish breaker block entry"
                            ))
                            break
                
                elif breaker.type == 'bearish':
                    # Price returning to bearish breaker from below
                    if bar['high'] >= breaker.low and bar['high'] <= breaker.high:
                        entry_price = bar['close']
                        stop_loss = breaker.high + atr.iloc[i] * 0.5
                        
                        # Find recent low for targets
                        recent_low = None
                        for sp in reversed(swing_points):
                            if sp.type == 'low' and sp.index > df.index[breaker.broken_idx]:
                                recent_low = sp
                                break
                        
                        if recent_low:
                            tp_range = breaker.low - recent_low.price
                            tp1 = entry_price - tp_range * 0.5
                            tp2 = entry_price - tp_range
                            tp3 = entry_price - tp_range * 1.5
                            
                            patterns.append(TradePattern(
                                pattern_type='breaker',
                                confidence=0.75,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=[tp1, tp2, tp3],
                                start_idx=breaker.start_idx,
                                end_idx=i,
                                direction='short',
                                notes="Bearish breaker block entry"
                            ))
                            break
        
        return patterns
    
    def find_fvg_patterns(self, df: pd.DataFrame,
                         fvgs: List,
                         market_structure: Dict) -> List[TradePattern]:
        """
        Find Fair Value Gap entry patterns
        
        Criteria:
        1. Price returns to FVG
        2. Aligns with market structure
        3. Shows reaction at FVG level
        """
        patterns = []
        atr = self.calculate_atr(df)
        
        for fvg in fvgs:
            if fvg.filled:
                continue
            
            # Look for price entering FVG
            for i in range(fvg.end_idx + 1, len(df)):
                bar = df.iloc[i]
                
                if fvg.type == 'bullish':
                    # Price entering bullish FVG from above
                    if bar['low'] <= fvg.high and bar['low'] >= fvg.low:
                        # Check market structure alignment
                        if market_structure.get('trend') != 'bearish':
                            entry_price = bar['close']
                            stop_loss = fvg.low - atr[i] * 0.5
                            
                            # Set targets
                            tp_range = abs(fvg.high - fvg.low) * 3
                            tp1 = entry_price + tp_range * 0.5
                            tp2 = entry_price + tp_range
                            tp3 = entry_price + tp_range * 1.5
                            
                            patterns.append(TradePattern(
                                pattern_type='fvg',
                                confidence=0.7,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=[tp1, tp2, tp3],
                                start_idx=fvg.start_idx,
                                end_idx=i,
                                direction='long',
                                notes="Bullish FVG entry"
                            ))
                            break
                
                elif fvg.type == 'bearish':
                    # Price entering bearish FVG from below
                    if bar['high'] >= fvg.low and bar['high'] <= fvg.high:
                        # Check market structure alignment
                        if market_structure.get('trend') != 'bullish':
                            entry_price = bar['close']
                            stop_loss = fvg.high + atr[i] * 0.5
                            
                            # Set targets
                            tp_range = abs(fvg.high - fvg.low) * 3
                            tp1 = entry_price - tp_range * 0.5
                            tp2 = entry_price - tp_range
                            tp3 = entry_price - tp_range * 1.5
                            
                            patterns.append(TradePattern(
                                pattern_type='fvg',
                                confidence=0.7,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=[tp1, tp2, tp3],
                                start_idx=fvg.start_idx,
                                end_idx=i,
                                direction='short',
                                notes="Bearish FVG entry"
                            ))
                            break
        
        return patterns
    
    def find_all_patterns(self, df: pd.DataFrame,
                         swing_points: List,
                         fvgs: List,
                         order_blocks: List,
                         breaker_blocks: List,
                         liquidity_levels: List,
                         market_structure: Dict) -> Dict[str, List[TradePattern]]:
        """Find all ICT patterns in the data"""
        all_patterns = {
            'ote': self.find_ote_patterns(df, swing_points, fvgs, order_blocks),
            'turtle_soup': self.find_turtle_soup_patterns(df, liquidity_levels, swing_points),
            'breaker': self.find_breaker_patterns(df, breaker_blocks, swing_points),
            'fvg': self.find_fvg_patterns(df, fvgs, market_structure)
        }
        
        return all_patterns