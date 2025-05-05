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

    def identify_order_blocks(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> List[OrderBlock]:
        order_blocks: List[OrderBlock] = []

        # Ensure index is unique to avoid get_loc returning slices
        if not df.index.is_unique:
            df = df.reset_index(drop=True)

        # Build a clean timestamp-to-row map
        idx_map = {ts: i for i, ts in enumerate(df.index)}

        for prev_sp, curr_sp in zip(swing_points[:-1], swing_points[1:]):
            try:
                prev_i = idx_map.get(prev_sp.index, None)
                curr_i = idx_map.get(curr_sp.index, None)

                if prev_i is None or curr_i is None:
                    continue
                if curr_i - prev_i < 2:
                    continue

                segment = df.iloc[prev_i:curr_i + 1]
                segment_cut = segment.iloc[:-1]

                # Bullish OB: price moves from low → high
                if prev_sp.type == "low" and curr_sp.type == "high":
                    reds = segment_cut[segment_cut["close"] < segment_cut["open"]]
                    if reds.empty:
                        continue

                    ob_row = reds.iloc[-1]
                    ob_loc = ob_row.name
                    move_strength = (curr_sp.price - prev_sp.price) / prev_sp.price

                    if move_strength >= self.min_ob_strength:
                        order_blocks.append(OrderBlock(
                            start_idx=ob_loc,
                            end_idx=ob_loc,
                            type="bullish",
                            high=ob_row["high"],
                            low=ob_row["low"],
                            open=ob_row["open"],
                            close=ob_row["close"],
                            mitigation_level=(ob_row["open"] + ob_row["close"]) / 2,
                            origin_swing="low"
                        ))

                # Bearish OB: price moves from high → low
                elif prev_sp.type == "high" and curr_sp.type == "low":
                    greens = segment_cut[segment_cut["close"] > segment_cut["open"]]
                    if greens.empty:
                        continue

                    ob_row = greens.iloc[-1]
                    ob_loc = ob_row.name
                    move_strength = (prev_sp.price - curr_sp.price) / prev_sp.price

                    if move_strength >= self.min_ob_strength:
                        order_blocks.append(OrderBlock(
                            start_idx=ob_loc,
                            end_idx=ob_loc,
                            type="bearish",
                            high=ob_row["high"],
                            low=ob_row["low"],
                            open=ob_row["open"],
                            close=ob_row["close"],
                            mitigation_level=(ob_row["open"] + ob_row["close"]) / 2,
                            origin_swing="high"
                        ))
            except Exception as e:
                print(f"[ERROR: OB loop] {e} for ob.start_idx={prev_sp.index}")

        return order_blocks


   

    
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
        for fvg in fvgs:
            for i in range(fvg.end_idx + 1, len(df)):
                candle = df.iloc[i]
                if fvg.type == 'bullish' and candle['low'] <= fvg.high:
                    fvg.filled = True
                    fvg.filled_idx = i
                    break
                elif fvg.type == 'bearish' and candle['high'] >= fvg.low:
                    fvg.filled = True
                    fvg.filled_idx = i
                    break
        
        return fvgs
    
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
        
        for i in range(1, len(swing_points)):
            current_swing = swing_points[i]
            
            # Check if this swing breaks previous structure
            for j in range(i):
                prev_swing = swing_points[j]
                
                # Bullish breaker: Break above previous high
                if current_swing.type == 'high' and prev_swing.type == 'high' and \
                   current_swing.price > prev_swing.price:
                    
                    # Find order blocks between these swings
                    relevant_obs = [ob for ob in order_blocks 
                                  if ob.start_idx > df.index.get_loc(prev_swing.index) and 
                                     ob.start_idx < df.index.get_loc(current_swing.index)]
                    
                    if relevant_obs:
                        # Last bearish OB becomes bullish breaker
                        bearish_obs = [ob for ob in relevant_obs if ob.type == 'bearish']
                        if bearish_obs:
                            last_bearish_ob = bearish_obs[-1]
                            breaker_blocks.append(BreakerBlock(
                                start_idx=last_bearish_ob.start_idx,
                                end_idx=last_bearish_ob.end_idx,
                                type='bullish',
                                high=last_bearish_ob.high,
                                low=last_bearish_ob.low,
                                origin_type='resistance',
                                broken_idx=df.index.get_loc(current_swing.index)
                            ))
                
                # Bearish breaker: Break below previous low
                elif current_swing.type == 'low' and prev_swing.type == 'low' and \
                     current_swing.price < prev_swing.price:
                    
                    # Find order blocks between these swings
                    relevant_obs = [ob for ob in order_blocks 
                                  if ob.start_idx > df.index.get_loc(prev_swing.index) and 
                                     ob.start_idx < df.index.get_loc(current_swing.index)]
                    
                    if relevant_obs:
                        # Last bullish OB becomes bearish breaker
                        bullish_obs = [ob for ob in relevant_obs if ob.type == 'bullish']
                        if bullish_obs:
                            last_bullish_ob = bullish_obs[-1]
                            breaker_blocks.append(BreakerBlock(
                                start_idx=last_bullish_ob.start_idx,
                                end_idx=last_bullish_ob.end_idx,
                                type='bearish',
                                high=last_bullish_ob.high,
                                low=last_bullish_ob.low,
                                origin_type='support',
                                broken_idx=df.index.get_loc(current_swing.index)
                            ))
        
        return breaker_blocks
    
    def identify_mitigation_blocks(self, df: pd.DataFrame, 
                                 order_blocks: List[OrderBlock]) -> List[Dict]:
        """
        Identify Mitigation Blocks
        
        A mitigation block occurs when price returns to an order block
        that has been "left behind" after a strong move
        
        Args:
            df: DataFrame with OHLCV data
            order_blocks: List of order blocks
            
        Returns:
            List of mitigation block dictionaries
        """
        mitigation_blocks = []
        
        for ob in order_blocks:
            # Check if price has returned to mitigate this order block
            for i in range(ob.end_idx + 1, len(df)):
                candle = df.iloc[i]
                
                if ob.type == 'bullish':
                    # Price returned to bullish OB from above
                    if candle['low'] <= ob.high and candle['low'] >= ob.low:
                        mitigation_blocks.append({
                            'order_block_idx': ob.start_idx,
                            'mitigation_idx': i,
                            'type': 'bullish',
                            'mitigation_price': candle['low'],
                            'original_ob': ob
                        })
                        break
                
                elif ob.type == 'bearish':
                    # Price returned to bearish OB from below
                    if candle['high'] >= ob.low and candle['high'] <= ob.high:
                        mitigation_blocks.append({
                            'order_block_idx': ob.start_idx,
                            'mitigation_idx': i,
                            'type': 'bearish',
                            'mitigation_price': candle['high'],
                            'original_ob': ob
                        })
                        break
        
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