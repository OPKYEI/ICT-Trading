# Trading Logic Documentation

## Overview

This document explains the trading logic implementation in ICT-ML-Trading, particularly the improved 5-bar window system that ensures alignment between model predictions and trade execution.

## The Problem We Solved

### Original Issue

The ML model is trained to answer: **"Will price be higher/lower 5 bars from now?"**

However, the original execution logic would:
1. Enter trades whenever the signal was generated
2. Hold positions until an opposite signal appeared
3. Re-enter trades after restarts regardless of price movement

This created a **fundamental misalignment**: The model expected trades to last exactly 5 bars, but execution allowed indefinite holding and late entries at unfavorable prices.

### Example Scenario

```
09:00 - Model says BUY (predicting price higher at 14:00)
09:00 - Enter BUY at 1.1000
09:15 - System crashes
09:30 - Restart system, price now at 1.1050 (50 pips higher)
09:30 - Original logic: Would BUY again at 1.1050 ❌
```

The model never trained on "buy after 50-pip move" scenarios!

## The Solution: 5-Bar Window Logic

### Core Principles

1. **Trades are valid for exactly 5 hours** (5 x 1-hour bars)
2. **No duplicate trades in the same direction within the window**
3. **Automatic position closure after 5 hours**
4. **Price validation for late entries**

### Implementation Details

#### 1. State Management

We maintain trade state in `trade_state.json`:

```json
{
  "active_trades": {
    "EUR_USD": {
      "signal": 1,
      "entry_time": "2024-01-15 09:00:00",
      "entry_price": 1.08500
    },
    "GBP_USD": {
      "signal": -1,
      "entry_time": "2024-01-15 08:00:00",
      "entry_price": 1.25000
    }
  },
  "last_check": "2024-01-15 10:00:00"
}
```

#### 2. Entry Logic

```python
def should_enter_trade(symbol, signal, current_price, current_time):
    trade = get_active_trade(symbol)
    
    # No active trade → Enter
    if not trade:
        return True
    
    # Trade expired (≥5 hours) → Enter
    if hours_elapsed(trade['entry_time']) >= 5:
        return True
    
    # Same direction within window → Skip
    if trade['signal'] == signal:
        return False
    
    # Different direction → Enter (after closing existing)
    return True
```

#### 3. Price Validation

For late entries (restarts), we validate price hasn't moved unfavorably:

```python
def is_price_valid_for_late_entry(symbol, signal, current_price):
    trade = get_active_trade(symbol)
    
    # BUY: Current price must be ≤ entry price
    if signal == 1:
        return current_price <= trade['entry_price']
    
    # SELL: Current price must be ≥ entry price
    elif signal == -1:
        return current_price >= trade['entry_price']
```

#### 4. Automatic Closure

Every hour, before processing new signals:

```python
def check_and_close_expired_trades():
    for symbol, trade in active_trades.items():
        if hours_elapsed(trade['entry_time']) >= 5:
            close_position(symbol)
            remove_from_state(symbol)
```

## Trading Flow Diagram

```
HOURLY EXECUTION FLOW
┌─────────────────────┐
│   System Starts     │
│   Load State File   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Close Expired      │
│  Trades (>5 hours)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Generate New       │
│  Signal for Symbol  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Check Entry Rules  │
├─────────────────────┤
│ • No active trade?  │
│ • Trade expired?    │
│ • Different signal? │
│ • Price still valid?│
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
  [ENTER]    [SKIP]
```

## Practical Examples

### Example 1: Normal Flow

```
09:00 - BUY signal → Enter BUY at 1.1000
10:00 - BUY signal → Skip (same direction, within window)
11:00 - BUY signal → Skip
12:00 - BUY signal → Skip
13:00 - BUY signal → Skip
14:00 - BUY signal → Auto-close previous trade, enter new BUY
```

### Example 2: Direction Change

```
09:00 - BUY signal → Enter BUY at 1.1000
10:00 - BUY signal → Skip
11:00 - SELL signal → Close BUY, enter SELL
12:00 - SELL signal → Skip
13:00 - SELL signal → Skip
14:00 - SELL signal → Skip
16:00 - SELL signal → Auto-close at 5 hours, enter new SELL
```

### Example 3: Restart Scenario

```
09:00 - BUY signal → Enter BUY at 1.1000
09:15 - System crashes
09:30 - Restart
        Price = 1.1020 (20 pips adverse)
        Check: 1.1020 > 1.1000
        Action: Skip (price validation failed)
10:00 - BUY signal → Skip (same trade active)
```

## Configuration Options

In the improved script, you can adjust:

```python
# Position size per trade
units = 50000  # 0.5 lots

# Window duration (hours)
WINDOW_HOURS = 5  # Matches model training

# Price tolerance for late entries (optional)
MAX_ADVERSE_PIPS = 10  # Could add this check
```

## Benefits of This Approach

1. **Model Alignment**: Execution matches training assumptions
2. **Risk Control**: No chasing extended moves
3. **Predictable Behavior**: Trades last exactly 5 hours
4. **Restart Resilience**: State persists across crashes
5. **Clean Exit**: Automatic position management

## Monitoring and Debugging

### Log Entries to Watch

```
INFO - EUR_USD: No active trade, can enter new position
INFO - EUR_USD: Same direction signal within 5-hour window (3.2h elapsed), skipping
INFO - EUR_USD: Previous trade expired (>5 hours), can enter new position
INFO - EUR_USD: Trade exceeded 5-hour window, closing position
WARNING - EUR_USD: Price moved too far from initial entry, skipping late entry
```

### State File Monitoring

Check `trade_state.json` to see:
- Active trades and their entry times
- Entry prices for validation
- Last system check time

### Verification Steps

1. Confirm trades close after exactly 5 hours
2. Verify no duplicate same-direction trades within window
3. Check restart behavior with state preservation
4. Monitor price validation rejections

## Migration Guide

To switch from original to improved logic:

1. **Stop current system**: `Ctrl+C`
2. **Close all positions manually** (if any)
3. **Delete old state** (if exists): `rm trade_state.json`
4. **Start improved system**: `python live_trade_multiinstrument_improved.py`
5. **Monitor first few trades** to verify behavior

## FAQ

**Q: What happens if I restart after 3 hours?**
A: System checks if price is still favorable. If yes, continues normally. If no, waits for next signal.

**Q: Can I change the 5-hour window?**
A: Not recommended. The model is trained on 5-bar predictions. Changing this requires retraining.

**Q: What if my broker doesn't report positions?**
A: The system uses its own state file as source of truth.

**Q: How does it handle partial fills?**
A: Current version assumes full fills. Partial fill handling could be added.

## Future Enhancements

1. **Dynamic Window Sizing**: Based on timeframe (5 bars for any TF)
2. **Partial Fill Handling**: Track actual vs intended position size
3. **Slippage Tracking**: Compare intended vs actual entry prices
4. **Multi-Timeframe Support**: Different windows for different timeframes

---

*The 5-bar window logic ensures your ML model's statistical edge is properly captured in live trading.*