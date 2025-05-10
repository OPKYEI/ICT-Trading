# src/features/liquidity.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LiquidityLevel:
    index: pd.Timestamp
    price: float
    type: str  # 'BSL' or 'SSL'
    strength: int
    swept: bool = False
    swept_index: Optional[pd.Timestamp] = None

@dataclass
class LiquidityPool:
    levels: List[LiquidityLevel]
    type: str
    high: float
    low: float
    center: float
    swept: bool = False
    swept_index: Optional[pd.Timestamp] = None

@dataclass
class StopRun:
    index: pd.Timestamp
    direction: str
    swept_level: float
    entry_price: float
    type: str
    levels_swept: List[LiquidityLevel]

class LiquidityAnalyzer:
    def __init__(self, 
                 lookback_period: int = 20,
                 liquidity_threshold: float = 0.0001,
                 pool_threshold: float = 0.0005):
        self.lookback_period = lookback_period
        self.liquidity_threshold = liquidity_threshold
        self.pool_threshold = pool_threshold

    def identify_liquidity_levels(self, df: pd.DataFrame, swing_points: List) -> List[LiquidityLevel]:
        liquidity_levels = []

        swing_highs = [sp for sp in swing_points if sp.type == 'high']
        swing_lows = [sp for sp in swing_points if sp.type == 'low']

        for swing in swing_highs:
            try:
                loc = df.index.get_loc(swing.index)
                if isinstance(loc, slice):
                    loc = loc.start or 0
                window_start = max(0, loc - self.lookback_period)
                window_end = loc #min(len(df), loc + self.lookback_period)
                window_data = df.iloc[window_start:window_end]
                touches = sum(abs(row['high'] - swing.price) / swing.price < self.liquidity_threshold
                              for _, row in window_data.iterrows())
                liquidity_levels.append(LiquidityLevel(swing.index, swing.price, 'BSL', touches))
            except Exception as e:
                print(f"[ERROR: BSL] {e} for swing.index={swing.index}")

        for swing in swing_lows:
            try:
                loc = df.index.get_loc(swing.index)
                if isinstance(loc, slice):
                    loc = loc.start or 0
                window_start = max(0, loc - self.lookback_period)
                window_end = loc #min(len(df), loc + self.lookback_period)
                window_data = df.iloc[window_start:window_end]
                touches = sum(abs(row['low'] - swing.price) / swing.price < self.liquidity_threshold
                              for _, row in window_data.iterrows())
                liquidity_levels.append(LiquidityLevel(swing.index, swing.price, 'SSL', touches))
            except Exception as e:
                print(f"[ERROR: SSL] {e} for swing.index={swing.index}")
        
        '''for level in liquidity_levels:
            try:
                loc = df.index.get_loc(level.index)
                if isinstance(loc, slice):
                    loc = loc.start or 0
                for i in range(loc + 1, len(df)):
                    if level.type == 'BSL' and df.iloc[i]['high'] > level.price:
                        level.swept = True
                        level.swept_index = df.index[i]
                        break
                    elif level.type == 'SSL' and df.iloc[i]['low'] < level.price:
                        level.swept = True
                        level.swept_index = df.index[i]
                        break
            except Exception as e:
                print(f"[ERROR: sweep check] {e} for level.index={level.index}")'''

        return liquidity_levels

    
    def identify_liquidity_pools(self,
                                 liquidity_levels: List[LiquidityLevel],
                                 cutoff_pos: Optional[int] = None) -> List[LiquidityPool]:
        """
        Group nearby liquidity levels into pools (backward-looking).

        Only levels observed at or before `cutoff_pos` are considered,
        so at bar t you never include future levels in your pools.
        Args:
            liquidity_levels: List of all liquidity levels detected so far
            cutoff_pos: maximum integer index to include (None => include all)
        Returns:
            List of LiquidityPool objects
        """
        pools: List[LiquidityPool] = []

        # 1) Filter out any levels beyond the current bar
        if cutoff_pos is not None:
            levels = [
                lvl for lvl in liquidity_levels
                if getattr(lvl, "start_idx", None) is None
                or lvl.start_idx <= cutoff_pos
            ]
        else:
            levels = liquidity_levels

        # 2) Group BSL (broken support levels)
        bsl_levels = [lvl for lvl in levels if lvl.type == "BSL"]
        bsl_levels.sort(key=lambda x: x.price)
        current_pool: List[LiquidityLevel] = []

        for lvl in bsl_levels:
            if not current_pool:
                current_pool.append(lvl)
                continue

            pool_low = min(l.price for l in current_pool)
            if (lvl.price - pool_low) / pool_low <= self.pool_threshold:
                current_pool.append(lvl)
            else:
                if len(current_pool) >= 2:
                    high = max(l.price for l in current_pool)
                    low  = pool_low
                    pools.append(LiquidityPool(
                        levels=current_pool.copy(),
                        type="BSL",
                        high=high,
                        low=low,
                        center=(high + low) / 2,
                        swept=all(l.swept for l in current_pool),
                        swept_index=max(
                            (l.swept_index for l in current_pool if l.swept_index),
                            default=None
                        )
                    ))
                current_pool = [lvl]

        # Final BSL pool
        if len(current_pool) >= 2:
            high = max(l.price for l in current_pool)
            low  = min(l.price for l in current_pool)
            pools.append(LiquidityPool(
                levels=current_pool.copy(),
                type="BSL",
                high=high,
                low=low,
                center=(high + low) / 2,
                swept=all(l.swept for l in current_pool),
                swept_index=max(
                    (l.swept_index for l in current_pool if l.swept_index),
                    default=None
                )
            ))

        # 3) Group SSL (broken support levels, reverse order)
        ssl_levels = [lvl for lvl in levels if lvl.type == "SSL"]
        ssl_levels.sort(key=lambda x: x.price, reverse=True)
        current_pool = []

        for lvl in ssl_levels:
            if not current_pool:
                current_pool.append(lvl)
                continue

            pool_high = max(l.price for l in current_pool)
            if (pool_high - lvl.price) / lvl.price <= self.pool_threshold:
                current_pool.append(lvl)
            else:
                if len(current_pool) >= 2:
                    high = pool_high
                    low  = min(l.price for l in current_pool)
                    pools.append(LiquidityPool(
                        levels=current_pool.copy(),
                        type="SSL",
                        high=high,
                        low=low,
                        center=(high + low) / 2,
                        swept=all(l.swept for l in current_pool),
                        swept_index=max(
                            (l.swept_index for l in current_pool if l.swept_index),
                            default=None
                        )
                    ))
                current_pool = [lvl]

        # Final SSL pool
        if len(current_pool) >= 2:
            high = max(l.price for l in current_pool)
            low  = min(l.price for l in current_pool)
            pools.append(LiquidityPool(
                levels=current_pool.copy(),
                type="SSL",
                high=high,
                low=low,
                center=(high + low) / 2,
                swept=all(l.swept for l in current_pool),
                swept_index=max(
                    (l.swept_index for l in current_pool if l.swept_index),
                    default=None
                )
            ))

        return pools

    
    def identify_stop_runs(self, df: pd.DataFrame, liquidity_levels: List[LiquidityLevel], reversal_threshold: float = 0.002) -> List[StopRun]:
        stop_runs = []

        for level in liquidity_levels:
            if not level.swept:
                continue

            sweep_loc = df.index.get_loc(level.swept_index)
            if isinstance(sweep_loc, slice):
                sweep_loc = sweep_loc.start or 0

            max_offset = min(6, len(df) - sweep_loc)
            for j in range(1, max_offset):
                current_bar = df.iloc[sweep_loc + j]
                sweep_bar = df.iloc[sweep_loc]

                if level.type == 'BSL':
                    if current_bar['close'] < sweep_bar['close'] - (sweep_bar['high'] - sweep_bar['low']):
                        reversal_size = (sweep_bar['high'] - current_bar['low']) / sweep_bar['high']
                        if reversal_size >= reversal_threshold:
                            swept_levels = [lvl for lvl in liquidity_levels if lvl.type == 'BSL' and lvl.swept_index == level.swept_index and lvl.price <= sweep_bar['high']]
                            stop_runs.append(StopRun(index=current_bar.name, direction='bearish', swept_level=sweep_bar['high'], entry_price=current_bar['close'], type='turtle_soup' if reversal_size >= reversal_threshold * 2 else 'stop_hunt', levels_swept=swept_levels))
                            break
                else:
                    if current_bar['close'] > sweep_bar['close'] + (sweep_bar['high'] - sweep_bar['low']):
                        reversal_size = (current_bar['high'] - sweep_bar['low']) / sweep_bar['low']
                        if reversal_size >= reversal_threshold:
                            swept_levels = [lvl for lvl in liquidity_levels if lvl.type == 'SSL' and lvl.swept_index == level.swept_index and lvl.price >= sweep_bar['low']]
                            stop_runs.append(StopRun(index=current_bar.name, direction='bullish', swept_level=sweep_bar['low'], entry_price=current_bar['close'], type='turtle_soup' if reversal_size >= reversal_threshold * 2 else 'stop_hunt', levels_swept=swept_levels))
                            break

        return stop_runs

    
    def classify_liquidity_run(self, df: pd.DataFrame, 
                             start_idx: int, end_idx: int,
                             liquidity_levels: List[LiquidityLevel]) -> Dict[str, any]:
        """
        Classify a price move as low or high resistance liquidity run
        
        Args:
            df: DataFrame with OHLCV data
            start_idx: Start index of the move
            end_idx: End index of the move
            liquidity_levels: List of liquidity levels
            
        Returns:
            Dictionary with classification details
        """
        start_price = df.iloc[start_idx]['close']
        end_price = df.iloc[end_idx]['close']
        
        # Determine direction
        direction = 'bullish' if end_price > start_price else 'bearish'
        
        # Find levels swept during this move
        swept_levels = []
        for level in liquidity_levels:
            if level.swept_index is not None:
                sweep_idx = df.index.get_loc(level.swept_index)
                if start_idx <= sweep_idx <= end_idx:
                    swept_levels.append(level)
        
        # Calculate resistance based on:
        # 1. Number of levels swept
        # 2. Strength of levels swept
        # 3. Speed of move (bars taken)
        # 4. Retracements during move
        
        total_strength = sum(level.strength for level in swept_levels)
        move_bars = end_idx - start_idx
        move_size = abs(end_price - start_price) / start_price
        
        # Calculate retracements
        retracements = 0
        max_retrace = 0
        if direction == 'bullish':
            current_high = start_price
            for i in range(start_idx + 1, end_idx + 1):
                if df.iloc[i]['high'] > current_high:
                    current_high = df.iloc[i]['high']
                else:
                    retrace = (current_high - df.iloc[i]['low']) / current_high
                    if retrace > 0.001:  # Significant retracement
                        retracements += 1
                        max_retrace = max(max_retrace, retrace)
        else:
            current_low = start_price
            for i in range(start_idx + 1, end_idx + 1):
                if df.iloc[i]['low'] < current_low:
                    current_low = df.iloc[i]['low']
                else:
                    retrace = (df.iloc[i]['high'] - current_low) / current_low
                    if retrace > 0.001:  # Significant retracement
                        retracements += 1
                        max_retrace = max(max_retrace, retrace)
        
        # Low resistance criteria:
        # - Few levels swept
        # - Low total strength
        # - Fast move (few bars)
        # - Minimal retracements
        
        resistance_score = (
            len(swept_levels) * 0.3 +
            total_strength * 0.3 +
            (move_bars / 10) * 0.2 +
            retracements * 0.2
        )
        
        classification = 'low_resistance' if resistance_score < 2.0 else 'high_resistance'
        
        return {
            'classification': classification,
            'direction': direction,
            'swept_levels': len(swept_levels),
            'total_strength': total_strength,
            'move_bars': move_bars,
            'move_size': move_size,
            'retracements': retracements,
            'max_retrace': max_retrace,
            'resistance_score': resistance_score
        }