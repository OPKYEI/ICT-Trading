# src/features/intermarket.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

@dataclass
class CorrelationResult:
    """Represents correlation analysis results"""
    correlation_coefficient: float
    p_value: float
    window: int
    is_significant: bool
    direction: str  # 'positive', 'negative', 'neutral'

@dataclass
class SMTDivergence:
    """Represents Smart Money Technique divergence"""
    asset1: str
    asset2: str
    divergence_type: str  # 'bullish' or 'bearish'
    start_idx: int
    end_idx: int
    asset1_direction: str  # 'higher_high', 'lower_high', etc.
    asset2_direction: str
    strength: float  # 0-1 scale
    confirmed: bool

@dataclass
class StrengthScore:
    """Represents currency strength score"""
    currency: str
    score: float
    rank: int
    trend: str  # 'strengthening', 'weakening', 'stable'
    momentum: float

class IntermarketAnalysis:
    """
    Implements ICT intermarket analysis concepts:
    - Dollar correlation analysis
    - SMT (Smart Money Technique) divergences
    - Currency strength analysis
    - Cross-pair correlations
    - Market confluences
    """
    
    def __init__(self, 
                 correlation_window: int = 20,
                 significance_level: float = 0.05,
                 min_divergence_bars: int = 5):
        """
        Args:
            correlation_window: Rolling window for correlation calculation
            significance_level: P-value threshold for statistical significance
            min_divergence_bars: Minimum bars for valid divergence
        """
        self.correlation_window = correlation_window
        self.significance_level = significance_level
        self.min_divergence_bars = min_divergence_bars
        
        # ICT standard forex pairs for analysis
        self.major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
                           'AUDUSD', 'USDCAD', 'NZDUSD']
        
        # Dollar index components and weights (ICE methodology)
        self.dxy_weights = {
            'EUR': 0.576,
            'JPY': 0.136,
            'GBP': 0.119,
            'CAD': 0.091,
            'SEK': 0.042,
            'CHF': 0.036
        }
    
    def calculate_correlation(self, 
                            series1: pd.Series, 
                            series2: pd.Series,
                            window: int = None) -> CorrelationResult:
        """
        Calculate rolling correlation between two series
        
        Args:
            series1: First price series
            series2: Second price series
            window: Correlation window (default: self.correlation_window)
            
        Returns:
            CorrelationResult object
        """
        if window is None:
            window = self.correlation_window
        
        # Ensure series are aligned
        aligned_data = pd.concat([series1, series2], axis=1).dropna()
        s1 = aligned_data.iloc[:, 0]
        s2 = aligned_data.iloc[:, 1]
        
        # Calculate correlation
        if len(s1) < window:
            raise ValueError(f"Insufficient data for correlation calculation. Need at least {window} points")
        
        # Rolling correlation
        rolling_corr = s1.rolling(window).corr(s2)
        latest_corr = rolling_corr.iloc[-1]
        
        # Statistical test
        if len(s1) >= window:
            corr_coef, p_value = stats.pearsonr(
                s1.iloc[-window:], 
                s2.iloc[-window:]
            )
        else:
            corr_coef, p_value = latest_corr, 1.0
        
        # Determine direction
        if abs(corr_coef) < 0.3:
            direction = 'neutral'
        elif corr_coef > 0:
            direction = 'positive'
        else:
            direction = 'negative'
        
        return CorrelationResult(
            correlation_coefficient=corr_coef,
            p_value=p_value,
            window=window,
            is_significant=p_value < self.significance_level,
            direction=direction
        )
    
    def detect_smt_divergence(self, 
                            asset1_data: pd.DataFrame,
                            asset2_data: pd.DataFrame,
                            lookback: int = 20) -> List[SMTDivergence]:
        """
        Detect Smart Money Technique divergences between two assets
        
        ICT SMT Concept:
        - When correlated assets move differently at key levels
        - Indicates potential reversals or continuation
        
        Args:
            asset1_data: OHLC data for first asset
            asset2_data: OHLC data for second asset
            lookback: Bars to look back for divergence
            
        Returns:
            List of SMTDivergence objects
        """
        divergences = []
        
        # Ensure data alignment
        common_index = asset1_data.index.intersection(asset2_data.index)
        df1 = asset1_data.loc[common_index]
        df2 = asset2_data.loc[common_index]
        
        # Find swing highs and lows
        for i in range(lookback, len(df1)):
            # Get recent price action
            asset1_segment = df1.iloc[i-lookback:i+1]
            asset2_segment = df2.iloc[i-lookback:i+1]
            
            # Check for divergence at highs
            asset1_highs = self._find_swing_highs(asset1_segment)
            asset2_highs = self._find_swing_highs(asset2_segment)
            
            if len(asset1_highs) >= 2 and len(asset2_highs) >= 2:
                # Compare last two highs
                a1_hh = asset1_highs[-1]['price'] > asset1_highs[-2]['price']
                a2_hh = asset2_highs[-1]['price'] > asset2_highs[-2]['price']
                
                # Bearish divergence: Asset1 makes HH, Asset2 makes LH
                if a1_hh and not a2_hh:
                    divergences.append(SMTDivergence(
                        asset1='Asset1',
                        asset2='Asset2',
                        divergence_type='bearish',
                        start_idx=asset1_highs[-2]['index'],
                        end_idx=asset1_highs[-1]['index'],
                        asset1_direction='higher_high',
                        asset2_direction='lower_high',
                        strength=self._calculate_divergence_strength(
                            asset1_highs[-2:], asset2_highs[-2:]
                        ),
                        confirmed=True
                    ))
                
                # Bullish divergence at lows
                asset1_lows = self._find_swing_lows(asset1_segment)
                asset2_lows = self._find_swing_lows(asset2_segment)
                
                if len(asset1_lows) >= 2 and len(asset2_lows) >= 2:
                    a1_ll = asset1_lows[-1]['price'] < asset1_lows[-2]['price']
                    a2_ll = asset2_lows[-1]['price'] < asset2_lows[-2]['price']
                    
                    # Bullish divergence: Asset1 makes LL, Asset2 makes HL
                    if a1_ll and not a2_ll:
                        divergences.append(SMTDivergence(
                            asset1='Asset1',
                            asset2='Asset2',
                            divergence_type='bullish',
                            start_idx=asset1_lows[-2]['index'],
                            end_idx=asset1_lows[-1]['index'],
                            asset1_direction='lower_low',
                            asset2_direction='higher_low',
                            strength=self._calculate_divergence_strength(
                                asset1_lows[-2:], asset2_lows[-2:]
                            ),
                            confirmed=True
                        ))
        
        return divergences
    
    def calculate_currency_strength(self, 
                                  pairs_data: Dict[str, pd.DataFrame],
                                  lookback: int = 14) -> Dict[str, StrengthScore]:
        """
        Calculate relative currency strength based on ICT concepts
        
        Args:
            pairs_data: Dictionary with pair names as keys and OHLC data as values
            lookback: Period for strength calculation
            
        Returns:
            Dictionary of currency strength scores
        """
        currencies = set()
        for pair in pairs_data.keys():
            currencies.add(pair[:3])
            currencies.add(pair[3:6])
        
        strength_scores = {}
        
        for currency in currencies:
            total_score = 0
            count = 0
            momentum = 0
            
            for pair, data in pairs_data.items():
                if currency in pair:
                    # Calculate performance
                    returns = data['close'].pct_change(lookback).iloc[-1]
                    
                    # If currency is base (first 3 letters)
                    if pair[:3] == currency:
                        score = returns * 100
                    # If currency is quote (last 3 letters)
                    else:
                        score = -returns * 100
                    
                    total_score += score
                    count += 1
                    
                    # Calculate momentum
                    short_returns = data['close'].pct_change(lookback // 2).iloc[-1]
                    if pair[:3] == currency:
                        momentum += short_returns * 100
                    else:
                        momentum += -short_returns * 100
            
            if count > 0:
                avg_score = total_score / count
                avg_momentum = momentum / count
                
                # Determine trend
                if avg_momentum > avg_score:
                    trend = 'strengthening'
                elif avg_momentum < avg_score:
                    trend = 'weakening'
                else:
                    trend = 'stable'
                
                strength_scores[currency] = StrengthScore(
                    currency=currency,
                    score=avg_score,
                    rank=0,  # Will be set after sorting
                    trend=trend,
                    momentum=avg_momentum
                )
        
        # Rank currencies
        sorted_currencies = sorted(
            strength_scores.items(), 
            key=lambda x: x[1].score, 
            reverse=True
        )
        
        for rank, (currency, score) in enumerate(sorted_currencies, 1):
            score.rank = rank
        
        return strength_scores
    
    def find_correlated_pairs(self, 
                            pairs_data: Dict[str, pd.DataFrame],
                            min_correlation: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find correlated and inversely correlated pairs
        
        Args:
            pairs_data: Dictionary of pair data
            min_correlation: Minimum absolute correlation threshold
            
        Returns:
            Dictionary with pair correlations
        """
        correlations = {}
        pairs = list(pairs_data.keys())
        
        for i in range(len(pairs)):
            pair1 = pairs[i]
            correlations[pair1] = []
            
            for j in range(i + 1, len(pairs)):
                pair2 = pairs[j]
                
                # Calculate correlation
                corr_result = self.calculate_correlation(
                    pairs_data[pair1]['close'],
                    pairs_data[pair2]['close']
                )
                
                if abs(corr_result.correlation_coefficient) >= min_correlation:
                    correlations[pair1].append(
                        (pair2, corr_result.correlation_coefficient)
                    )
                    
                    # Add reverse correlation
                    if pair2 not in correlations:
                        correlations[pair2] = []
                    correlations[pair2].append(
                        (pair1, corr_result.correlation_coefficient)
                    )
        
        return correlations
    
    def calculate_dollar_index_proxy(self, 
                                   pairs_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate Dollar Index proxy using ICE methodology
        
        Args:
            pairs_data: Dictionary with major pairs data
            
        Returns:
            Series representing dollar index proxy
        """
        # Required pairs for DXY calculation
        required_pairs = {
            'EUR': 'EURUSD',
            'JPY': 'USDJPY',
            'GBP': 'GBPUSD',
            'CAD': 'USDCAD',
            'CHF': 'USDCHF'
        }
        
        # Check if we have required pairs
        missing_pairs = []
        for currency, pair in required_pairs.items():
            if pair not in pairs_data:
                missing_pairs.append(pair)
        
        if missing_pairs:
            raise ValueError(f"Missing required pairs for DXY calculation: {missing_pairs}")
        
        # Get common index
        common_index = pairs_data[required_pairs['EUR']].index
        for pair in required_pairs.values():
            common_index = common_index.intersection(pairs_data[pair].index)
        
        # Calculate weighted index
        dxy_proxy = pd.Series(index=common_index, dtype=float)
        dxy_proxy[:] = 100.0  # Base value
        
        for currency, pair in required_pairs.items():
            if currency in ['EUR', 'GBP']:
                # Inverse pairs (USD is quote)
                component = 1 / pairs_data[pair].loc[common_index, 'close']
            else:
                # Direct pairs (USD is base)
                component = pairs_data[pair].loc[common_index, 'close']
            
            # Apply weight
            weight = self.dxy_weights.get(currency, 0)
            dxy_proxy *= component ** weight
        
        return dxy_proxy
    
    def _find_swing_highs(self, data: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find swing highs in price data"""
        highs = []
        
        for i in range(window, len(data) - window):
            if data['high'].iloc[i] == data['high'].iloc[i-window:i+window+1].max():
                highs.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'timestamp': data.index[i]
                })
        
        return highs
    
    def _find_swing_lows(self, data: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find swing lows in price data"""
        lows = []
        
        for i in range(window, len(data) - window):
            if data['low'].iloc[i] == data['low'].iloc[i-window:i+window+1].min():
                lows.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'timestamp': data.index[i]
                })
        
        return lows
    
    def _calculate_divergence_strength(self, 
                                     swings1: List[Dict], 
                                     swings2: List[Dict]) -> float:
        """Calculate divergence strength (0-1 scale)"""
        if len(swings1) < 2 or len(swings2) < 2:
            return 0.0
        
        # Calculate price change percentages
        change1 = abs(swings1[-1]['price'] - swings1[-2]['price']) / swings1[-2]['price']
        change2 = abs(swings2[-1]['price'] - swings2[-2]['price']) / swings2[-2]['price']
        
        # Divergence strength is based on the difference in changes
        divergence_magnitude = abs(change1 - change2)
        
        # Normalize to 0-1 scale (using 5% as max expected divergence)
        strength = min(divergence_magnitude / 0.05, 1.0)
        
        return strength