# src/features/pd_arrays.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .market_structure import SwingPoint

@dataclass
class OrderBlock:
    """Represents an order block"""
    start_idx: int
    end_idx: int
    type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    open: float
    close: float
    mitigation_level: float  # 50% of the order block
    origin_swing: Optional[str] = None  # 'high' or 'low' that created this OB

@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (FVG)"""
    start_idx: int
    end_idx: int
    type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    filled: bool = False
    filled_idx: Optional[int] = None

@dataclass
class BreakerBlock:
    """Represents a breaker block"""
    start_idx: int
    end_idx: int
    type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    origin_type: str  # original 'support' or 'resistance'
    broken_idx: int  # index where structure was broken



class PriceDeliveryArrays:
    """
    Identifies ICT Price Delivery Arrays including:
    - Order Blocks (Bullish/Bearish)
    - Fair Value Gaps (Imbalances)
    - Breaker Blocks
    - Mitigation Blocks
    - Rejection Blocks
    """

    def __init__(self, min_order_block_strength: float = 0.001):
        """
        Args:
            min_order_block_strength: Minimum price movement % to qualify as order block
        """
        self.min_ob_strength = min_order_block_strength
    def identify_order_blocks(self, df: pd.DataFrame,
                              swing_points: List[SwingPoint]) -> List[OrderBlock]:
        """
        Identify Order Blocks in a backward‐looking manner.

        At bar t, only swings up to t are considered. Returns the list
        of all order blocks (start/end indices are absolute positions).
        """
        obs: List[OrderBlock] = []
        # Map index → integer position
        idx_map = {ts: i for i, ts in enumerate(df.index)}
        # Only consider swing points up to the last available index
        max_pos = len(df) - 1

        # Walk through swing_points sequentially
        for i in range(1, len(swing_points)):
            prev, curr = swing_points[i - 1], swing_points[i]
            try:
                prev_pos = idx_map[prev.index]
                curr_pos = idx_map[curr.index]
            except KeyError:
                # swing outside our df index—skip
                continue

            # Ensure we never use swings beyond current df length
            if prev_pos > max_pos or curr_pos > max_pos:
                continue

            # Bearish OB: swing high → swing low
            if prev.type == 'high' and curr.type == 'low':
                high = df.iloc[prev_pos]['high']
                low  = df.iloc[curr_pos]['low']
                obs.append(OrderBlock(
                    start_idx=prev_pos,
                    end_idx=curr_pos,
                    type='bearish',
                    high=high,
                    low=low,
                    open=df.iloc[prev_pos]['open'],
                    close=df.iloc[prev_pos]['close'],
                    mitigation_level=(high + low) / 2,
                    origin_swing=prev.type
                ))

            # Bullish OB: swing low → swing high
            elif prev.type == 'low' and curr.type == 'high':
                low  = df.iloc[prev_pos]['low']
                high = df.iloc[curr_pos]['high']
                obs.append(OrderBlock(
                    start_idx=prev_pos,
                    end_idx=curr_pos,
                    type='bullish',
                    high=high,
                    low=low,
                    open=df.iloc[prev_pos]['open'],
                    close=df.iloc[prev_pos]['close'],
                    mitigation_level=(high + low) / 2,
                    origin_swing=prev.type
                ))

        return obs

    def identify_breaker_blocks(self, df: pd.DataFrame, 
                           order_blocks: List[OrderBlock],
                           swing_points: List) -> List[BreakerBlock]:
        """
        Identify Breaker Blocks
        
        A breaker block forms when:
        1. Price breaks a swing high/low
        2. The last order block before the break becomes a breaker
        
        Args:
            df: DataFrame with OHLCV data
            order_blocks: List of identified order blocks
            swing_points: List of swing points
            
        Returns:
            List of BreakerBlock objects
        """
        breaker_blocks = []
        
        # Ensure df index is unique to avoid get_loc duplicates
        if not df.index.is_unique:
            df = df.reset_index(drop=True)
        
        # Create a mapping from index value to position
        idx_map = {ts: i for i, ts in enumerate(df.index)}
        
        # Process each swing point
        for i in range(1, len(swing_points)):
            current_swing = swing_points[i]
            current_idx = current_swing.index
            current_pos = idx_map.get(current_idx, -1)
            if current_pos == -1:
                continue
            
            # Check if this swing breaks previous structure
            for j in range(i):
                prev_swing = swing_points[j]
                prev_idx = prev_swing.index  
                prev_pos = idx_map.get(prev_idx, -1)
                if prev_pos == -1:
                    continue
                
                # Bullish breaker: Break above previous high
                if current_swing.type == 'high' and prev_swing.type == 'high' and \
                   current_swing.price > prev_swing.price:
                    
                    # Find order blocks between these swings - USING POSITIONS
                    relevant_obs = []
                    for ob in order_blocks:
                        # Convert timestamp start_idx to position if needed
                        ob_pos = ob.start_idx
                        if isinstance(ob_pos, pd.Timestamp):
                            ob_pos = idx_map.get(ob_pos, -1)
                            if ob_pos == -1:
                                continue
                        
                        # Compare integer positions
                        if ob_pos > prev_pos and ob_pos < current_pos:
                            relevant_obs.append(ob)
                    
                    if relevant_obs:
                        # Last bearish OB becomes bullish breaker
                        bearish_obs = [ob for ob in relevant_obs if ob.type == 'bearish']
                        if bearish_obs:
                            last_bearish_ob = bearish_obs[-1]
                            # Ensure the broken_idx is an integer position
                            broken_pos = current_pos
                            breaker_blocks.append(BreakerBlock(
                                start_idx=last_bearish_ob.start_idx,
                                end_idx=last_bearish_ob.end_idx,
                                type='bullish',
                                high=last_bearish_ob.high,
                                low=last_bearish_ob.low,
                                origin_type='resistance',
                                broken_idx=broken_pos
                            ))
                
                # Bearish breaker: Break below previous low
                elif current_swing.type == 'low' and prev_swing.type == 'low' and \
                     current_swing.price < prev_swing.price:
                    
                    # Find order blocks between these swings - USING POSITIONS
                    relevant_obs = []
                    for ob in order_blocks:
                        # Convert timestamp start_idx to position if needed
                        ob_pos = ob.start_idx
                        if isinstance(ob_pos, pd.Timestamp):
                            ob_pos = idx_map.get(ob_pos, -1)
                            if ob_pos == -1:
                                continue
                        
                        # Compare integer positions
                        if ob_pos > prev_pos and ob_pos < current_pos:
                            relevant_obs.append(ob)
                    
                    if relevant_obs:
                        # Last bullish OB becomes bearish breaker
                        bullish_obs = [ob for ob in relevant_obs if ob.type == 'bullish']
                        if bullish_obs:
                            last_bullish_ob = bullish_obs[-1]
                            # Ensure the broken_idx is an integer position
                            broken_pos = current_pos
                            breaker_blocks.append(BreakerBlock(
                                start_idx=last_bullish_ob.start_idx,
                                end_idx=last_bullish_ob.end_idx,
                                type='bearish',
                                high=last_bullish_ob.high,
                                low=last_bullish_ob.low,
                                origin_type='support',
                                broken_idx=broken_pos
                            ))
        
        return breaker_blocks


   

    
    def identify_fvg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Identify Fair Value Gaps (FVG) / Imbalances
        
        FVG occurs when there's a gap between:
        - Candle 1 high and Candle 3 low (bearish FVG)
        - Candle 1 low and Candle 3 high (bullish FVG)
        With Candle 2 showing strong directional movement
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of FairValueGap objects
        """
        fvgs = []
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: Gap up
            if candle3['low'] > candle1['high']:
                # Verify candle 2 is bullish and has momentum
                if candle2['close'] > candle2['open'] and \
                   (candle2['close'] - candle2['open']) > abs(candle1['close'] - candle1['open']):
                    fvgs.append(FairValueGap(
                        start_idx=i-2,
                        end_idx=i,
                        type='bullish',
                        high=candle3['low'],
                        low=candle1['high'],
                        filled=False
                    ))
            
            # Bearish FVG: Gap down
            elif candle3['high'] < candle1['low']:
                # Verify candle 2 is bearish and has momentum
                if candle2['close'] < candle2['open'] and \
                   (candle2['open'] - candle2['close']) > abs(candle1['close'] - candle1['open']):
                    fvgs.append(FairValueGap(
                        start_idx=i-2,
                        end_idx=i,
                        type='bearish',
                        high=candle1['low'],
                        low=candle3['high'],
                        filled=False
                    ))
        
        # Check if FVGs have been filled
        '''for fvg in fvgs:
            for i in range(fvg.end_idx + 1, len(df)):
                candle = df.iloc[i]
                if fvg.type == 'bullish' and candle['low'] <= fvg.high:
                    fvg.filled = True
                    fvg.filled_idx = i
                    break
                elif fvg.type == 'bearish' and candle['high'] >= fvg.low:
                    fvg.filled = True
                    fvg.filled_idx = i
                    break '''
        
        return fvgs
    
    def identify_breaker_blocks(self, df: pd.DataFrame, 
                               order_blocks: List[OrderBlock],
                               swing_points: List[SwingPoint]) -> List[BreakerBlock]:
        """
        Identify Breaker Blocks
        
        A breaker block forms when:
        1. Price breaks a swing high/low
        2. The last order block before the break becomes a breaker
        
        Args:
            df: DataFrame with OHLCV data
            order_blocks: List of identified order blocks
            swing_points: List of swing points
            
        Returns:
            List of BreakerBlock objects
        """
        breaker_blocks = []
        
        # First separate bullish and bearish order blocks
        bullish_order_blocks = [ob for ob in order_blocks if ob.type == 'bullish']
        bearish_order_blocks = [ob for ob in order_blocks if ob.type == 'bearish']
        
        # Create a mapping from timestamps to positions for reliable lookups
        ts_to_pos = {ts: idx for idx, ts in enumerate(df.index)}
        
        # Process bullish breakers
        bullish_breakers = []
        for ob in bearish_order_blocks:
            start_idx = ob.start_idx
            if isinstance(start_idx, pd.Timestamp):
                start_pos = ts_to_pos.get(start_idx, -1)
            else:
                start_pos = start_idx
            if start_pos == -1:
                continue
                
            for bulls in bullish_order_blocks:
                bulls_idx = bulls.start_idx
                if isinstance(bulls_idx, pd.Timestamp):
                    bulls_pos = ts_to_pos.get(bulls_idx, -1)
                else:
                    bulls_pos = bulls_idx
                if bulls_pos == -1:
                    continue
                    
                if start_pos < bulls_pos:
                    if bulls.high >= ob.high:
                        breaker_blocks.append(BreakerBlock(
                            start_idx=bulls.start_idx,
                            end_idx=bulls.end_idx,
                            type='bullish',
                            high=bulls.high,
                            low=bulls.low,
                            origin_type='resistance',
                            broken_idx=bulls_pos
                        ))
                        bullish_breakers.append(bulls)
        
        # Process bearish breakers
        bearish_breakers = []
        for ob in bullish_order_blocks:
            start_idx = ob.start_idx
            if isinstance(start_idx, pd.Timestamp):
                start_pos = ts_to_pos.get(start_idx, -1)
            else:
                start_pos = start_idx
            if start_pos == -1:
                continue
                
            for bears in bearish_order_blocks:
                bears_idx = bears.start_idx
                if isinstance(bears_idx, pd.Timestamp):
                    bears_pos = ts_to_pos.get(bears_idx, -1)
                else:
                    bears_pos = bears_idx
                if bears_pos == -1:
                    continue
                    
                if start_pos < bears_pos:
                    if bears.low <= ob.low:
                        breaker_blocks.append(BreakerBlock(
                            start_idx=bears.start_idx,
                            end_idx=bears.end_idx,
                            type='bearish',
                            high=bears.high,
                            low=bears.low,
                            origin_type='support',
                            broken_idx=bears_pos
                        ))
                        bearish_breakers.append(bears)
        
        return breaker_blocks

    
    def identify_mitigation_blocks(self,
                                   df: pd.DataFrame,
                                   order_blocks: List[OrderBlock]) -> List[Dict]:
        """
        Identify Mitigation Blocks (backward‐looking)

        A mitigation block occurs the first time price returns into the original
        order-block range, _after_ that block formed.  At each bar t we only
        consider candles at or before t.
        """
        # Map timestamps → positions for fast lookup
        idx_map = {ts: i for i, ts in enumerate(df.index)}
        # Track which OBs are still unmitigated
        unmitigated = {ob: True for ob in order_blocks}
        mitigation_blocks: List[Dict] = []

        # Walk forward through the entire series
        for t, (ts, candle) in enumerate(df.iterrows()):
            # Check every still-open order block
            for ob in order_blocks:
                if not unmitigated.get(ob):
                    continue

                # Determine the integer position of this OB’s end
                ob_pos = ob.end_idx if isinstance(ob.end_idx, int) \
                         else idx_map.get(ob.end_idx, None)
                if ob_pos is None or t <= ob_pos:
                    # either we haven't reached the block’s end yet,
                    # or no valid pos → skip
                    continue

                # Bullish OB: price must dip down into the block’s high→low
                if ob.type == 'bullish':
                    if candle['low'] <= ob.high and candle['low'] >= ob.low:
                        mitigation_blocks.append({
                            'order_block_idx': ob.start_idx,
                            'mitigation_idx': t,
                            'type': 'bullish',
                            'mitigation_price': float(candle['low']),
                            'original_ob': ob
                        })
                        unmitigated[ob] = False

                # Bearish OB: price must rally up into the block’s low→high
                elif ob.type == 'bearish':
                    if candle['high'] >= ob.low and candle['high'] <= ob.high:
                        mitigation_blocks.append({
                            'order_block_idx': ob.start_idx,
                            'mitigation_idx': t,
                            'type': 'bearish',
                            'mitigation_price': float(candle['high']),
                            'original_ob': ob
                        })
                        unmitigated[ob] = False

        return mitigation_blocks
    
    def identify_rejection_blocks(self, df: pd.DataFrame, 
                                swing_points: List) -> List[Dict]:
        """
        Identify Rejection Blocks
        
        A rejection block forms when price creates a wick (rejection)
        at a significant level, typically at swing points
        
        Args:
            df: DataFrame with OHLCV data
            swing_points: List of swing points
            
        Returns:
            List of rejection block dictionaries
        """
        rejection_blocks = []
        
        for swing in swing_points:
            swing_idx = df.index.get_loc(swing.index)
            candle = df.iloc[swing_idx]
            
            # Calculate wick sizes
            if swing.type == 'high':
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                body_size = abs(candle['close'] - candle['open'])
                
                # Significant upper wick indicates rejection
                if upper_wick > body_size * 1.5:  # Wick is 1.5x larger than body
                    rejection_blocks.append({
                        'index': swing_idx,
                        'type': 'bearish',
                        'rejection_price': candle['high'],
                        'body_high': max(candle['open'], candle['close']),
                        'body_low': min(candle['open'], candle['close']),
                        'wick_size': upper_wick
                    })
            
            elif swing.type == 'low':
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                body_size = abs(candle['close'] - candle['open'])
                
                # Significant lower wick indicates rejection
                if lower_wick > body_size * 1.5:  # Wick is 1.5x larger than body
                    rejection_blocks.append({
                        'index': swing_idx,
                        'type': 'bullish',
                        'rejection_price': candle['low'],
                        'body_high': max(candle['open'], candle['close']),
                        'body_low': min(candle['open'], candle['close']),
                        'wick_size': lower_wick
                    })
        
        return rejection_blocks