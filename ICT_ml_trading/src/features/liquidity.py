# src/features/liquidity.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LiquidityLevel:
    """Represents a liquidity level (high or low)"""
    index: pd.Timestamp
    price: float
    type: str  # 'BSL' (Buy-side) or 'SSL' (Sell-side)
    strength: int  # Number of touches/tests
    swept: bool = False
    swept_index: Optional[pd.Timestamp] = None

@dataclass
class LiquidityPool:
    """Represents a pool of liquidity (multiple levels close together)"""
    levels: List[LiquidityLevel]
    type: str  # 'BSL' or 'SSL'
    high: float
    low: float
    center: float
    swept: bool = False
    swept_index: Optional[pd.Timestamp] = None

@dataclass
class StopRun:
    """Represents a stop run event"""
    index: pd.Timestamp
    direction: str  # 'bullish' or 'bearish'
    swept_level: float
    entry_price: float
    type: str  # 'turtle_soup' or 'stop_hunt'
    levels_swept: List[LiquidityLevel]

class LiquidityAnalyzer:
    """
    Analyzes liquidity in the market following ICT concepts:
    - Buy-side liquidity (BSL) and Sell-side liquidity (SSL)
    - Liquidity pools and voids
    - Stop runs and turtle soup patterns
    - Low resistance vs high resistance liquidity runs
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 liquidity_threshold: float = 0.0001,
                 pool_threshold: float = 0.0005):
        """
        Args:
            lookback_period: Bars to look back for liquidity levels
            liquidity_threshold: Minimum price difference for separate levels
            pool_threshold: Maximum price difference to group levels into pools
        """
        self.lookback_period = lookback_period
        self.liquidity_threshold = liquidity_threshold
        self.pool_threshold = pool_threshold
    
    def identify_liquidity_levels(self, df: pd.DataFrame, 
                                swing_points: List) -> List[LiquidityLevel]:
        """
        Identify significant highs and lows that represent liquidity
        
        Args:
            df: DataFrame with OHLCV data
            swing_points: List of swing highs and lows
            
        Returns:
            List of LiquidityLevel objects
        """
        liquidity_levels = []
        
        # Extract highs and lows from swing points
        swing_highs = [sp for sp in swing_points if sp.type == 'high']
        swing_lows = [sp for sp in swing_points if sp.type == 'low']
        
        # Process swing highs for BSL
        for swing in swing_highs:
            # Check how many times this level was tested
            window_start = max(0, df.index.get_loc(swing.index) - self.lookback_period)
            window_end = min(len(df), df.index.get_loc(swing.index) + self.lookback_period)
            window_data = df.iloc[window_start:window_end]
            
            # Count touches (price came within threshold)
            touches = 0
            for idx, row in window_data.iterrows():
                if abs(row['high'] - swing.price) / swing.price < self.liquidity_threshold:
                    touches += 1
            
            liquidity_levels.append(LiquidityLevel(
                index=swing.index,
                price=swing.price,
                type='BSL',
                strength=touches
            ))
        
        # Process swing lows for SSL
        for swing in swing_lows:
            # Check how many times this level was tested
            window_start = max(0, df.index.get_loc(swing.index) - self.lookback_period)
            window_end = min(len(df), df.index.get_loc(swing.index) + self.lookback_period)
            window_data = df.iloc[window_start:window_end]
            
            # Count touches
            touches = 0
            for idx, row in window_data.iterrows():
                if abs(row['low'] - swing.price) / swing.price < self.liquidity_threshold:
                    touches += 1
            
            liquidity_levels.append(LiquidityLevel(
                index=swing.index,
                price=swing.price,
                type='SSL',
                strength=touches
            ))
        
        # Check if levels have been swept
        for level in liquidity_levels:
            level_idx = df.index.get_loc(level.index)
            
            # Check bars after the level was formed
            for i in range(level_idx + 1, len(df)):
                if level.type == 'BSL' and df.iloc[i]['high'] > level.price:
                    level.swept = True
                    level.swept_index = df.index[i]
                    break
                elif level.type == 'SSL' and df.iloc[i]['low'] < level.price:
                    level.swept = True
                    level.swept_index = df.index[i]
                    break
        
        return liquidity_levels
    
    def identify_liquidity_pools(self, 
                               liquidity_levels: List[LiquidityLevel]) -> List[LiquidityPool]:
        """
        Group nearby liquidity levels into pools
        
        Args:
            liquidity_levels: List of identified liquidity levels
            
        Returns:
            List of LiquidityPool objects
        """
        pools = []
        bsl_levels = [lvl for lvl in liquidity_levels if lvl.type == 'BSL']
        ssl_levels = [lvl for lvl in liquidity_levels if lvl.type == 'SSL']
        
        # Group BSL levels
        bsl_levels.sort(key=lambda x: x.price)
        current_pool_levels = []
        
        for level in bsl_levels:
            if not current_pool_levels:
                current_pool_levels.append(level)
            else:
                # Check if close enough to current pool
                pool_high = max(lvl.price for lvl in current_pool_levels)
                pool_low = min(lvl.price for lvl in current_pool_levels)
                
                if (level.price - pool_low) / pool_low <= self.pool_threshold:
                    current_pool_levels.append(level)
                else:
                    # Create pool from current levels
                    if len(current_pool_levels) >= 2:
                        pool_high = max(lvl.price for lvl in current_pool_levels)
                        pool_low = min(lvl.price for lvl in current_pool_levels)
                        pools.append(LiquidityPool(
                            levels=current_pool_levels.copy(),
                            type='BSL',
                            high=pool_high,
                            low=pool_low,
                            center=(pool_high + pool_low) / 2,
                            swept=all(lvl.swept for lvl in current_pool_levels),
                            swept_index=max((lvl.swept_index for lvl in current_pool_levels 
                                           if lvl.swept_index), default=None)
                        ))
                    current_pool_levels = [level]
        
        # Create final BSL pool if needed
        if len(current_pool_levels) >= 2:
            pool_high = max(lvl.price for lvl in current_pool_levels)
            pool_low = min(lvl.price for lvl in current_pool_levels)
            pools.append(LiquidityPool(
                levels=current_pool_levels.copy(),
                type='BSL',
                high=pool_high,
                low=pool_low,
                center=(pool_high + pool_low) / 2,
                swept=all(lvl.swept for lvl in current_pool_levels),
                swept_index=max((lvl.swept_index for lvl in current_pool_levels 
                               if lvl.swept_index), default=None)
            ))
        
        # Group SSL levels (similar process)
        ssl_levels.sort(key=lambda x: x.price, reverse=True)
        current_pool_levels = []
        
        for level in ssl_levels:
            if not current_pool_levels:
                current_pool_levels.append(level)
            else:
                pool_high = max(lvl.price for lvl in current_pool_levels)
                pool_low = min(lvl.price for lvl in current_pool_levels)
                
                if (pool_high - level.price) / level.price <= self.pool_threshold:
                    current_pool_levels.append(level)
                else:
                    if len(current_pool_levels) >= 2:
                        pool_high = max(lvl.price for lvl in current_pool_levels)
                        pool_low = min(lvl.price for lvl in current_pool_levels)
                        pools.append(LiquidityPool(
                            levels=current_pool_levels.copy(),
                            type='SSL',
                            high=pool_high,
                            low=pool_low,
                            center=(pool_high + pool_low) / 2,
                            swept=all(lvl.swept for lvl in current_pool_levels),
                            swept_index=max((lvl.swept_index for lvl in current_pool_levels 
                                           if lvl.swept_index), default=None)
                        ))
                    current_pool_levels = [level]
        
        # Create final SSL pool if needed
        if len(current_pool_levels) >= 2:
            pool_high = max(lvl.price for lvl in current_pool_levels)
            pool_low = min(lvl.price for lvl in current_pool_levels)
            pools.append(LiquidityPool(
                levels=current_pool_levels.copy(),
                type='SSL',
                high=pool_high,
                low=pool_low,
                center=(pool_high + pool_low) / 2,
                swept=all(lvl.swept for lvl in current_pool_levels),
                swept_index=max((lvl.swept_index for lvl in current_pool_levels 
                               if lvl.swept_index), default=None)
            ))
        
        return pools
    
    def identify_stop_runs(self, df: pd.DataFrame, 
                          liquidity_levels: List[LiquidityLevel],
                          reversal_threshold: float = 0.002) -> List[StopRun]:
        """
        Identify stop runs and turtle soup patterns
        
        Args:
            df: DataFrame with OHLCV data
            liquidity_levels: List of liquidity levels
            reversal_threshold: Minimum reversal size to confirm stop run
            
        Returns:
            List of StopRun objects
        """
        stop_runs = []
        
        for i, level in enumerate(liquidity_levels):
            if not level.swept:
                continue
                
            sweep_idx = df.index.get_loc(level.swept_index)
            
            # Check for reversal after sweep
            if level.type == 'BSL':
                # Bullish stop run - price sweeps BSL then reverses down
                # Look for reversal within next 5 bars
                for j in range(1, min(6, len(df) - sweep_idx)):
                    current_bar = df.iloc[sweep_idx + j]
                    sweep_bar = df.iloc[sweep_idx]
                    
                    # Check if price reversed down significantly
                    if current_bar['close'] < sweep_bar['close'] - (sweep_bar['high'] - sweep_bar['low']):
                        # Calculate reversal size
                        reversal_size = (sweep_bar['high'] - current_bar['low']) / sweep_bar['high']
                        
                        if reversal_size >= reversal_threshold:
                            # Find all levels swept in this run
                            swept_levels = []
                            for other_level in liquidity_levels:
                                if (other_level.type == 'BSL' and 
                                    other_level.swept_index == level.swept_index and
                                    other_level.price <= sweep_bar['high']):
                                    swept_levels.append(other_level)
                            
                            stop_runs.append(StopRun(
                                index=current_bar.name,
                                direction='bearish',
                                swept_level=sweep_bar['high'],
                                entry_price=current_bar['close'],
                                type='turtle_soup' if reversal_size >= reversal_threshold * 2 else 'stop_hunt',
                                levels_swept=swept_levels
                            ))
                            break
            
            else:  # SSL
                # Bearish stop run - price sweeps SSL then reverses up
                for j in range(1, min(6, len(df) - sweep_idx)):
                    current_bar = df.iloc[sweep_idx + j]
                    sweep_bar = df.iloc[sweep_idx]
                    
                    # Check if price reversed up significantly
                    if current_bar['close'] > sweep_bar['close'] + (sweep_bar['high'] - sweep_bar['low']):
                        # Calculate reversal size
                        reversal_size = (current_bar['high'] - sweep_bar['low']) / sweep_bar['low']
                        
                        if reversal_size >= reversal_threshold:
                            # Find all levels swept in this run
                            swept_levels = []
                            for other_level in liquidity_levels:
                                if (other_level.type == 'SSL' and 
                                    other_level.swept_index == level.swept_index and
                                    other_level.price >= sweep_bar['low']):
                                    swept_levels.append(other_level)
                            
                            stop_runs.append(StopRun(
                                index=current_bar.name,
                                direction='bullish',
                                swept_level=sweep_bar['low'],
                                entry_price=current_bar['close'],
                                type='turtle_soup' if reversal_size >= reversal_threshold * 2 else 'stop_hunt',
                                levels_swept=swept_levels
                            ))
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