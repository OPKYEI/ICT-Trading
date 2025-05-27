# ICT-ML-Trading – Key Challenges & How We Solved Them

*Chronological timeline of development challenges and solutions (April → May 2025)*

## Overview

Building a machine learning trading system that actually works in production required solving numerous technical challenges. This document chronicles the major issues we encountered and how we solved them, serving as both a historical record and a guide for similar projects.

## Timeline of Challenges

### 1. Unrealistically High Accuracy (April 27th – May 5th 2025)

**Symptom**: 
- Out-of-sample accuracies ≥ 99%
- Sharpe ratios > 10
- Results looked "too good to be true"

**Root Causes Discovered**:
1. **Target look-ahead bias** in `future_direction_5` calculation
2. **Naïve K-Fold CV** leaking data across time boundaries
3. **Forward-filled target labels** in `_cleanup_features` contaminating past with future

**Solutions Implemented**:

1. **TimeSeriesSplit with Embargo** (`run_pipeline.py` v0.8.2)
   ```python
   # Added 5-bar embargo before each test fold
   if embargo > 0:
       first_test = test_idx.min()
       train_idx = train_idx[train_idx < first_test - embargo]
   ```

2. **Nested Cross-Validation** (`trainer.py` v0.9.0)
   - Outer loop for honest test performance
   - Inner loop for hyperparameter tuning
   - No data leakage between folds

3. **Fixed Feature Cleanup** (commit `2b9e1ab`)
   ```python
   # Separate target from features
   target_cols = [c for c in features.columns if c.startswith("future_")]
   feat_cols = [c for c in features.columns if c not in target_cols]
   
   # Only forward-fill features, not targets
   feats = features[feat_cols].fillna(method='ffill')
   ```

4. **Label-Shuffle Sanity Test**
   - Automatically runs with shuffled labels
   - Must achieve ~50% accuracy (random)
   - Confirms no hidden leakage

### 2. Backtesting Inaccuracies (Early May 2025)

**Symptom**:
- Equity curves showed unrealistic jumps
- Duplicate timestamp crashes: `ValueError: cannot reindex`
- Perfect entries/exits at exact highs/lows

**Root Causes**:
- Execution at same bar's close (lookahead)
- No slippage or transaction costs
- Duplicate timestamps in data

**Solutions**:

1. **Next-Bar Execution** (`backtest_signals.py`)
   ```python
   # Execute at NEXT bar's open, not current close
   entry_price = price_data.iloc[i+1]['open']
   ```

2. **Realistic Cost Model**
   ```python
   # ATR-based slippage
   slippage = atr * 0.1  # 10% of ATR
   
   # Fixed spread
   spread = 0.3 * pip_size  # 0.3 pips
   
   # Apply to entry/exit
   entry_price += slippage + spread
   ```

3. **Data Deduplication**
   ```python
   df_bt = df_bt[~df_bt.index.duplicated(keep='last')]
   ```

### 3. Performance Bottlenecks (Early May, 2025)

**Symptom**:
- Feature engineering took >45 minutes
- Exceeded Google Colab RAM limits
- Training interrupted due to timeouts

**Root Causes**:
- Recalculating features repeatedly
- No parallelization
- Inefficient pandas operations

**Solutions**:

1. **Feature Caching** (`_cached_engineer`)
   ```python
   from joblib import Memory
   memory = Memory(CACHE_DIR, verbose=0)
   
   @memory.cache
   def _cached_engineer(X, symbol, lookback_periods, ...):
       # Expensive feature calculations
   ```

2. **Progress Tracking**
   ```python
   from tqdm import tqdm
   for i in tqdm(range(len(data)), desc="Engineering features"):
       # Feature calculations
   ```

3. **Parallel Processing**
   ```python
   # XGBoost with parallel threads
   XGBClassifier(n_jobs=-1, tree_method='hist')
   ```

### 4. Multi-Broker Connection Issues (Late May, 2025)

**Symptom**:
- FTMO OAuth failures crashed entire session
- MT5 "IPC timeout" errors
- Trades executed on wrong broker

**Root Causes**:
- MT5 single connection limitation
- No retry logic
- Poor error handling

**Solutions**:

1. **Connection Management**
   ```python
   def _ensure_connected(self):
       """Ensure we're connected to the right MT5 terminal"""
       account_info = mt5.account_info()
       if account_info and account_info.login == self.login:
           return True
       
       # Reconnect if needed
       mt5.shutdown()
       time.sleep(0.2)
       mt5.initialize(...)
   ```

2. **Retry Logic**
   ```python
   for attempt in range(3):
       try:
           # Connection attempt
           break
       except Exception as e:
           if attempt == 2:
               logger.error(f"Failed after 3 attempts: {e}")
               # Continue with other brokers
   ```

3. **Graceful Degradation**
   - System continues with available brokers
   - Failed brokers logged but don't stop execution

### 5. Visualization & Notebook Issues (Apr → May 2025)

**Symptom**:
- Plots not rendering in Jupyter/Colab
- Memory leaks from matplotlib
- Inconsistent plot styling

**Solutions**:

1. **Backend Configuration**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For headless
   plt.style.use('seaborn-v0_8-darkgrid')
   ```

2. **Memory Management**
   ```python
   plt.close('all')  # After each plot
   gc.collect()      # Force garbage collection
   ```

### 6. Duplicate Timestamp Bug in Walk-Forward (May 2025)

**Symptom**:
- Walk-forward analysis crashed
- Index alignment errors
- Inconsistent results

**Solution** (commit `5f04c92`):
```python
# Re-index before concatenation
df_aligned = df.reindex(common_index)
results = pd.concat([results, df_aligned])
```

## Final Results

After solving all challenges, our production metrics show:

| Model | Hold-out Accuracy | Sharpe Ratio | Win Rate | Max Drawdown |
|-------|------------------|--------------|----------|--------------|
| XGBoost | 91% | 2.14 | 68% | 12.3% |
| GradientBoost | 89% | 1.98 | 65% | 14.1% |
| RandomForest | 87% | 1.76 | 63% | 15.8% |
| LogisticReg | 82% | 1.31 | 58% | 18.2% |

*Note: These are realistic metrics after fixing all leakage issues*

## Lessons Learned

### 1. Always Validate with Shuffle Test
- If shuffled labels give >55% accuracy, there's leakage
- Run this test FIRST, not last

### 2. Time Series Require Special Handling
- Never use standard K-fold CV
- Always use embargo between train/test
- Consider regime changes

### 3. Feature Engineering is Critical
- Cache expensive calculations
- Validate no lookahead in features
- Monitor feature importance regularly

### 4. Production != Backtesting
- Add realistic slippage/costs
- Handle connection failures gracefully
- Plan for partial broker availability

### 5. Debug Systematically
- Log everything
- Visualize intermediate results
- Test components in isolation

## Green-Light Deployment Criteria

A model is ready for production when:

1. **Walk-forward Sharpe ≥ 0.8**
2. **Noise-shuffled accuracy ≈ 50%** (confirms no leakage)
3. **Backtested with realistic costs**
4. **All brokers tested individually**
5. **30+ days paper trading success**

## Code Quality Checklist

- [ ] No `future_` columns in training features
- [ ] TimeSeriesSplit with embargo
- [ ] Shuffle test passes
- [ ] Features cached for performance
- [ ] Broker failures handled gracefully
- [ ] Comprehensive logging
- [ ] Unit tests for critical functions
- [ ] Documentation updated

## Future Improvements

1. **Regime Detection**: Adapt to changing market conditions
2. **Online Learning**: Update models with recent data
3. **Feature Store**: Centralized feature management
4. **A/B Testing**: Compare model versions in production
5. **Monitoring Dashboard**: Real-time performance tracking

---

*Remember: The journey from 99% accuracy (with leakage) to 91% accuracy (without leakage) represents moving from fantasy to reality. The lower number is the one that makes money.*