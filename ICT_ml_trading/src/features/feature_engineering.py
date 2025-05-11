# src/features/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from .market_structure import MarketStructureAnalyzer
from .pd_arrays import PriceDeliveryArrays
from .liquidity import LiquidityAnalyzer
from .time_features import TimeFeatures
from .patterns import PatternRecognition
from .intermarket import IntermarketAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
logger = logging.getLogger(__name__)
from pathlib import Path
from joblib import Memory

# Set up cache directory (adjust path as needed)
CACHE_DIR = Path(__file__).parent.parent / "cache" / "feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

@memory.cache
def _cached_engineer(
    X: pd.DataFrame,
    symbol: str,
    lookback_periods: tuple,
    feature_selection_threshold: float,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    additional_data: dict
) -> pd.DataFrame:
    """
    Standalone cached feature-engineering function (no self).
    """
    # 1) Build the engineer with the same params
    fe = ICTFeatureEngineer(
        lookback_periods=list(lookback_periods),
        feature_selection_threshold=feature_selection_threshold
    )
    # 2) Engineer features
    fs = fe.engineer_features(
        data=X,
        symbol=symbol,
        additional_data=additional_data
    )
    feats = fs.features.copy()
    # 3) Drop all look-ahead columns
    drop_cols = [c for c in feats.columns if c.startswith("future_")]
    feats.drop(columns=drop_cols, inplace=True, errors="ignore")
    # 4) Re-index to match X exactly
    return feats.reindex(X.index)

@dataclass
class FeatureSet:
    """Container for engineered features"""
    features: pd.DataFrame
    metadata: Dict[str, Any]
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None

class ICTFeatureEngineer:
    """
    Combines all ICT features into a comprehensive feature set for ML models.
    
    Features include:
    - Market structure (HH/HL/LH/LL, trends, MSS)
    - PD Arrays (order blocks, FVGs, breakers)
    - Liquidity (levels, pools, stop runs)
    - Time-based features (sessions, kill zones)
    - Pattern recognition (OTE, turtle soup)
    - Intermarket analysis (correlations, divergences)
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [5, 10, 20, 50],
                 feature_selection_threshold: float = 0.01):
        """
        Args:
            lookback_periods: Periods for calculating moving features
            feature_selection_threshold: Minimum importance for feature selection
        """
        self.lookback_periods = lookback_periods
        self.feature_selection_threshold = feature_selection_threshold
        
        # Initialize analyzers
        self.market_structure = MarketStructureAnalyzer()
        self.pd_arrays = PriceDeliveryArrays()
        self.liquidity = LiquidityAnalyzer()
        self.time_features = TimeFeatures()
        self.patterns = PatternRecognition()
        self.intermarket = IntermarketAnalysis()
        
        # Feature configuration
        self.feature_config = {
            'market_structure': True,
            'pd_arrays': True,
            'liquidity': True,
            'time_features': True,
            'patterns': True,
            'intermarket': True,
            'technical_indicators': True
        }
    
    def engineer_features(self, 
                      data: pd.DataFrame,
                      symbol: str,
                      additional_data: Optional[Dict[str, pd.DataFrame]] = None) -> FeatureSet:
        print(f"🔧 Engineering features for {symbol}")

        features = pd.DataFrame(index=data.index)
        feature_metadata = {
            'symbol': symbol,
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'total_bars': len(data)
        }
        
        if self.feature_config['market_structure']:
            print("→ Adding market structure features")
            features = self._add_market_structure_features(data, features)

        if self.feature_config['time_features']:
            print("→ Adding time features")
            features = self._add_time_features(data, features)

        if self.feature_config['pd_arrays']:
            print("→ Adding PD array features")
            features = self._add_pd_array_features(data, features)

        if self.feature_config['liquidity']:
            print("→ Adding liquidity features")
            features = self._add_liquidity_features(data, features)

        if self.feature_config['patterns']:
            print("→ Adding pattern features")
            features = self._add_pattern_features(data, features)

        if self.feature_config['intermarket'] and additional_data:
            print("→ Adding intermarket features")
            features = self._add_intermarket_features(data, features, additional_data)

        if self.feature_config['technical_indicators']:
            print("→ Adding technical indicators")
            features = self._add_technical_indicators(data, features)

        print("→ Adding target variable")
        features = self._add_target_variables(data, features)

        print("→ Cleaning up features")
        features = self._cleanup_features(features)

        print("✅ Feature engineering complete")

        return FeatureSet(
            features=features,
            metadata=feature_metadata,
            feature_names=list(features.columns)
        )

    
    def _add_market_structure_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features"""
        # Identify swing points
        swing_points = self.market_structure.identify_swing_points(data)
        
        # Initialize features
        features['swing_high'] = 0
        features['swing_low'] = 0
        features['swing_high_price'] = np.nan
        features['swing_low_price'] = np.nan
        
        # Mark swing points
        for sp in swing_points:
            if sp.type == 'high':
                features.loc[sp.index, 'swing_high'] = 1
                features.loc[sp.index, 'swing_high_price'] = sp.price
            else:
                features.loc[sp.index, 'swing_low'] = 1
                features.loc[sp.index, 'swing_low_price'] = sp.price
        
        # Forward fill swing prices
        features['last_swing_high'] = features['swing_high_price'].ffill()
        features['last_swing_low'] = features['swing_low_price'].ffill()

        
        # Distance from swings
        features['distance_from_swing_high'] = (data['close'] - features['last_swing_high']) / features['last_swing_high']
        features['distance_from_swing_low'] = (data['close'] - features['last_swing_low']) / features['last_swing_low']
        
        # Market structure classification
        structure = self.market_structure.classify_market_structure(swing_points)
        features['market_trend'] = structure.trend
        features['market_trend_encoded'] = features['market_trend'].map({
            'bullish': 1, 'bearish': -1, 'ranging': 0
        })
        
        # Structure strength
        if structure.trend == 'bullish' and structure.last_hh and structure.last_hl:
            features['structure_strength'] = abs(structure.last_hh.price - structure.last_hl.price) / structure.last_hl.price
        elif structure.trend == 'bearish' and structure.last_lh and structure.last_ll:
            features['structure_strength'] = abs(structure.last_lh.price - structure.last_ll.price) / structure.last_lh.price
        else:
            features['structure_strength'] = 0
        
        # MSS detection
        mss_result = self.market_structure.detect_market_structure_shift(data, swing_points, structure)
        features['mss_detected'] = 0
        features['bos_detected'] = 0
        
        if mss_result['mss_detected'] and mss_result['shift_index'] in features.index:
            features.loc[mss_result['shift_index'], 'mss_detected'] = 1
        if mss_result['bos_detected'] and mss_result['shift_index'] in features.index:
            features.loc[mss_result['shift_index'], 'bos_detected'] = 1
        
        return features
    
    def _add_time_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features using rolling/expanding windows to avoid lookahead.
        - Hourly highs/lows computed over a 1-hour window ending at current bar.
        - Daily ranges computed over a 1-day window ending at current bar.
        """
        idx = data.index

        # Basic time fields
        features['hour'] = idx.hour
        features['day_of_week'] = idx.dayofweek

        # Sessions
        features['in_london_session']   = features['hour'].between(7, 15)
        features['in_new_york_session'] = features['hour'].between(12, 20)
        features['in_asian_session']    = (features['hour'] >= 23) | (features['hour'] < 8)
        features['is_optimal_time']     = False  # placeholder

        # 1️⃣ Hourly rolling high & low (no future data)
        features['hour_high'] = (
            data['high']
            .rolling('1H', closed='both')  # 1-hour window up to current timestamp
            .max()
            .reindex(idx)
        )
        features['hour_low'] = (
            data['low']
            .rolling('1H', closed='both')
            .min()
            .reindex(idx)
        )
        features['hour_range'] = features['hour_high'] - features['hour_low']

        # 2️⃣ Daily rolling range up to current timestamp
        daily_high = (
            data['high']
            .rolling('1D', closed='both')  # 1-day window up to current timestamp
            .max()
            .reindex(idx)
        )
        daily_low = (
            data['low']
            .rolling('1D', closed='both')
            .min()
            .reindex(idx)
        )
        daily_range = daily_high - daily_low

        # 3️⃣ Session-specific daily ranges
        features['london_range']  = daily_range.where(features['in_london_session'],   other=0.0)
        features['newyork_range'] = daily_range.where(features['in_new_york_session'], other=0.0)
        features['asia_range']    = daily_range.where(features['in_asian_session'],    other=0.0)

        return features
   
    def _add_pd_array_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        swing_points = self.market_structure.identify_swing_points(data)

        order_blocks = self.pd_arrays.identify_order_blocks(data, swing_points)
        features['near_bullish_ob'] = 0
        features['near_bearish_ob'] = 0

        for ob in order_blocks:
            try:
                # Validate what ob.start_idx is
                if isinstance(ob.start_idx, (pd.Timestamp, str)):
                    ob_idx = data.index.get_loc(ob.start_idx)
                elif isinstance(ob.start_idx, (int, np.integer)):
                    ob_idx = ob.start_idx
                else:
                    print(f"Invalid ob.start_idx: {ob.start_idx} of type {type(ob.start_idx)}")
                    continue

                # Define window safely
                end_idx = min(ob_idx + 10, len(data))

                for i in range(ob_idx + 1, end_idx):
                    row = data.iloc[i]
                    if ob.type == 'bullish':
                        if row['low'] <= ob.high and row['low'] >= ob.low:
                            features.at[features.index[i], 'near_bullish_ob'] = 1
                    elif ob.type == 'bearish':
                        if row['high'] >= ob.low and row['high'] <= ob.high:
                            features.at[features.index[i], 'near_bearish_ob'] = 1
            except Exception as e:
                print(f"[ERROR: OB loop] {e} for ob.start_idx={ob.start_idx}")
                continue

        return features



        # --- Fair Value Gaps ---
        fvgs = self.pd_arrays.identify_fvg(data)
        features['fvg_bullish'] = 0
        features['fvg_bearish'] = 0
        for fvg in fvgs:
            col = 'fvg_bullish' if fvg.type == 'bullish' else 'fvg_bearish'
            for i in range(fvg.start_idx, fvg.end_idx + 1):
                if i < len(features):
                    features.iloc[i, features.columns.get_loc(col)] = 1

        # --- Breaker Blocks ---
        breakers = self.pd_arrays.identify_breaker_blocks(data, order_blocks, swing_points)
        features['breaker_bullish'] = 0
        features['breaker_bearish'] = 0
        for bb in breakers:
            col = 'breaker_bullish' if bb.type == 'bullish' else 'breaker_bearish'
            if bb.start_idx < len(features):
                features.iloc[bb.start_idx, features.columns.get_loc(col)] = 1

        # --- Mitigation Blocks ---
        mitigations = self.pd_arrays.identify_mitigation_blocks(data, order_blocks)
        features['mitigation_bullish'] = 0
        features['mitigation_bearish'] = 0
        for m in mitigations:
            col = 'mitigation_bullish' if m['type'] == 'bullish' else 'mitigation_bearish'
            idx = m['mitigation_idx']
            if idx < len(features):
                features.iloc[idx, features.columns.get_loc(col)] = 1

        # --- Rejection Blocks ---
        rejections = self.pd_arrays.identify_rejection_blocks(data, swing_points)
        features['rejection_bullish'] = 0
        features['rejection_bearish'] = 0
        for r in rejections:
            col = 'rejection_bullish' if r['type'] == 'bullish' else 'rejection_bearish'
            idx = r['index']
            if idx < len(features):
                features.iloc[idx, features.columns.get_loc(col)] = 1

        return features

    
    def _add_liquidity_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity features"""
        # Get swing points for liquidity analysis
        swing_points = self.market_structure.identify_swing_points(data)
        
        # Liquidity levels
        liquidity_levels = self.liquidity.identify_liquidity_levels(data, swing_points)
        features['near_bsl'] = 0
        features['near_ssl'] = 0
        features['liquidity_distance'] = np.nan
        
        for i, row in data.iterrows():
            for level in liquidity_levels:
                distance = abs(row['close'] - level.price) / level.price
                if distance < 0.001:  # Within 0.1% of liquidity level
                    if level.type == 'BSL':
                        features.loc[i, 'near_bsl'] = 1
                    else:
                        features.loc[i, 'near_ssl'] = 1
                    features.loc[i, 'liquidity_distance'] = distance
        
        # Stop runs
        stop_runs = self.liquidity.identify_stop_runs(data, liquidity_levels)
        features['stop_run'] = 0
        features['stop_run_direction'] = 'none'
        
        for run in stop_runs:
            features.loc[run.index, 'stop_run'] = 1
            features.loc[run.index, 'stop_run_direction'] = run.direction
        
        # Liquidity pools
        pools = self.liquidity.identify_liquidity_pools(liquidity_levels)
        features['near_liquidity_pool'] = 0
        
        for i, row in data.iterrows():
            for pool in pools:
                if row['high'] >= pool.low and row['low'] <= pool.high:
                    features.loc[i, 'near_liquidity_pool'] = 1
        
        return features
    
    def _add_pattern_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features (strictly backward-looking)."""
        # 1️⃣ Gather all contexts
        swing_points    = self.market_structure.identify_swing_points(data)
        fvgs            = self.pd_arrays.identify_fvg(data) or []
        order_blocks    = self.pd_arrays.identify_order_blocks(data, swing_points) or []
        breaker_blocks  = self.pd_arrays.identify_breaker_blocks(data, order_blocks, swing_points) or []
        liquidity_lvls  = self.liquidity.identify_liquidity_levels(data, swing_points) or []

        # 2️⃣ Market‐structure context
        struct = self.market_structure.classify_market_structure(swing_points)
        mkt_ctx = {'trend': struct.trend}

        # 3️⃣ Discover patterns, ensuring we get dicts of lists (never None)
        raw_patterns = {}
        try:
            raw_patterns = self.patterns.find_all_patterns(
                data, swing_points, fvgs, order_blocks,
                breaker_blocks, liquidity_lvls, mkt_ctx
            ) or {}
        except Exception:
            raw_patterns = {}

        # Normalize: convert any None → []
        patterns: Dict[str, List] = {
            ptype: (plist if isinstance(plist, list) else [])
            for ptype, plist in raw_patterns.items()
        }

        # 4️⃣ Initialize feature columns
        features['pattern_detected']   = 0
        features['pattern_type']       = 'none'
        features['pattern_direction']  = 'none'
        features['pattern_confidence'] = 0.0

        # 5️⃣ Populate
        for ptype, plist in patterns.items():
            for pat in plist:
                # end_idx is strictly in the past relative to each bar
                if not hasattr(pat, 'end_idx'):
                    continue
                try:
                    ts = data.index[pat.end_idx]
                except (IndexError, KeyError):
                    continue
                features.loc[ts, 'pattern_detected']   = 1
                features.loc[ts, 'pattern_type']       = getattr(pat, 'pattern_type', ptype)
                features.loc[ts, 'pattern_direction']  = getattr(pat, 'direction', 'none')
                features.loc[ts, 'pattern_confidence'] = getattr(pat, 'confidence', 0.0)

        return features

    
    def _add_intermarket_features(self, data: pd.DataFrame, features: pd.DataFrame, 
                                additional_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add intermarket analysis features"""
        # Dollar correlation (if USD pairs)
        if 'EURUSD' in additional_data or 'DXY' in additional_data:
            dxy_data = additional_data.get('DXY', additional_data.get('EURUSD'))
            if dxy_data is not None:
                corr_result = self.intermarket.calculate_correlation(
                    data['close'], 
                    dxy_data['close'],
                    window=20
                )
                features['dollar_correlation'] = corr_result.correlation_coefficient
                features['dollar_corr_direction'] = corr_result.direction
        
        # Currency strength
        if len(additional_data) >= 3:
            strength_scores = self.intermarket.calculate_currency_strength(additional_data)
            
            # Extract base and quote currencies from symbol
            if len(data.name) >= 6:
                base_currency = data.name[:3]
                quote_currency = data.name[3:6]
                
                if base_currency in strength_scores:
                    features['base_currency_strength'] = strength_scores[base_currency].score
                    features['base_currency_rank'] = strength_scores[base_currency].rank
                
                if quote_currency in strength_scores:
                    features['quote_currency_strength'] = strength_scores[quote_currency].score
                    features['quote_currency_rank'] = strength_scores[quote_currency].rank
        
        # SMT divergences
        for symbol, other_data in additional_data.items():
            divergences = self.intermarket.detect_smt_divergence(data, other_data)
            features[f'smt_divergence_{symbol}'] = 0
            
            for div in divergences:
                features.loc[data.index[div.end_idx], f'smt_divergence_{symbol}'] = 1
        
        return features
    
    def _add_technical_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""

        # ATR for volatility
        features['atr'] = self.patterns.calculate_atr(data)
        features['atr_normalized'] = features['atr'] / data['close']

        # Moving averages
        for period in self.lookback_periods:
            features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            features[f'distance_from_sma_{period}'] = (
                data['close'] - features[f'sma_{period}']
            ) / features[f'sma_{period}']

        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        features['bb_middle'] = data['close'].rolling(window=20).mean()
        features['bb_std'] = data['close'].rolling(window=20).std()
        features['bb_upper'] = features['bb_middle'] + (features['bb_std'] * 2)
        features['bb_lower'] = features['bb_middle'] - (features['bb_std'] * 2)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (
            features['bb_upper'] - features['bb_lower']
        )

        # Volume features (optional)
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
        else:
            print("⚠️ Skipping volume indicators: 'volume' column not found in data.")

        return features

    
    def _add_target_variables(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add target variables for supervised learning"""
        # Future returns
        for period in [1, 5, 10, 20]:
            features[f'future_return_{period}'] = data['close'].pct_change(period).shift(-period)
            features[f'future_direction_{period}'] = (features[f'future_return_{period}'] > 0).astype(int)
        
        # Future high/low
        for period in [5, 10, 20]:
            features[f'future_high_{period}'] = data['high'].rolling(window=period).max().shift(-period)
            features[f'future_low_{period}'] = data['low'].rolling(window=period).min().shift(-period)
            
            # Maximum favorable/adverse excursion
            features[f'mfe_{period}'] = (features[f'future_high_{period}'] - data['close']) / data['close']
            features[f'mae_{period}'] = (data['close'] - features[f'future_low_{period}']) / data['close']
        
        return features
    #FUNCTIONAL BEFORE ATTEMPTING ALL CHANGES FOR LEAKAGE
    def _cleanup_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up features:
          1) Encode categoricals
          2) Forward-fill only non-target features
          3) Drop rows where any future_ target is NaN
          4) Remove infinities and any remaining NaNs
        """
        # 1) Handle categorical variables
        cat_cols = features.select_dtypes(include=['object']).columns
        for col in cat_cols:
            features[f'{col}_encoded'] = pd.factorize(features[col])[0]
        features = features.drop(columns=cat_cols)

        # 2) Separate targets vs. features
        target_cols = [c for c in features.columns if c.startswith("future_")]
        feat_cols   = [c for c in features.columns if c not in target_cols]

        # Forward-fill only the non-target features
        feats = features[feat_cols].fillna(method='ffill')

        # 3) Drop rows where any target is NaN
        targets = features[target_cols].dropna()

        # 4) Recombine: only rows with valid targets
        clean = pd.concat([feats, targets], axis=1, join='inner')

        # 5) Remove infinite values, then drop any new NaNs
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

        return clean

    
    def select_features(self, features: pd.DataFrame, target_column: str, 
                       method: str = 'importance') -> List[str]:
        """
        Select most important features.
        
        Args:
            features: DataFrame with all features
            target_column: Target variable for feature importance
            method: Selection method ('importance', 'correlation', 'both')
            
        Returns:
            List of selected feature names
        """
        if method == 'importance' or method == 'both':
            # Use Random Forest for feature importance
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            X = features.drop(columns=[col for col in features.columns if 'future_' in col])
            y = features[target_column]
            
            # Remove any remaining NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            
            # Get feature importance
            importance = pd.Series(rf.feature_importances_, index=X.columns)
            importance = importance.sort_values(ascending=False)
            
            # Select features above threshold
            selected_features = list(importance[importance > self.feature_selection_threshold].index)
        
        if method == 'correlation' or method == 'both':
            # Remove highly correlated features
            corr_matrix = features.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation > 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            
            if method == 'both':
                selected_features = [f for f in selected_features if f not in to_drop]
            else:
                selected_features = [f for f in features.columns if f not in to_drop and 'future_' not in f]
        
        return selected_features

class ICTFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that wraps your existing ICTFeatureEngineer
    and automatically drops any look-ahead “future_…” columns.
    """
    def __init__(
        self,
        lookback_periods=[5, 10, 20],
        feature_selection_threshold=0.01,
        symbol=None,
        additional_data=None
    ):
        # Simply store the constructor argument; do NOT mutate it
        self.lookback_periods = lookback_periods
        self.feature_selection_threshold = feature_selection_threshold
        self.symbol = symbol
        self.additional_data = additional_data

    def fit(self, X, y=None):
        # No fitting required for pure feature engineering
        return self

    def transform(self, X):
        # 1) Compute base features via cached engineer
        start_ts = X.index.min()
        end_ts   = X.index.max()
        lbp      = tuple(self.lookback_periods)
        fst      = self.feature_selection_threshold
        sym      = self.symbol or ""
        feats    = _cached_engineer(
            X, sym, lbp, fst, start_ts, end_ts, self.additional_data
        )

        # 2) Merge multi-timeframe series
        for tf, df_tf in self.additional_data.items():
            # Align the higher-timeframe data to X's index, then prefix columns
            df_aligned   = df_tf.reindex(X.index, method="ffill")
            df_prefixed  = df_aligned.add_prefix(f"{tf}_")
            # Join into the feature matrix
            feats = feats.join(df_prefixed, how="left")

        return feats


    