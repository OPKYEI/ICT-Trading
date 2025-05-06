# src/trading/backtester.py

"""
Backtesting framework: simulate equity curve from bar-by-bar signals,
with transaction costs, slippage, and next-bar execution.
"""
import pandas as pd

def backtest_signals(
    signals: pd.DataFrame,
    price: pd.DataFrame,
    initial_equity: float = 10_000.0,
    trade_cost: float = 0.0002,   # 0.02% round-trip cost
    slippage: float = 0.0001      # 0.01% per trade slippage
) -> pd.Series:
    """
    Args:
        signals: DataFrame with 'signal' column (1=long, -1=short, 0=flat)
        price:   DataFrame with 'close' prices
        initial_equity: starting capital
        trade_cost: pct cost per unit change in position (round‐trip)
        slippage:   pct slippage per trade
        
    Returns:
        equity_curve: pd.Series of equity over time
    """
    # 1️⃣ Alignment check
    if not signals.index.equals(price.index):
        raise KeyError("Signals and price index must align exactly")
    
    # 2️⃣ Build unified DataFrame
    df = pd.DataFrame(index=signals.index)
    df['signal'] = signals['signal']
    df['close']  = price['close']
    
    # 3️⃣ Compute bar returns (close_t / close_{t-1} - 1)
    df['return'] = df['close'].pct_change().fillna(0)
    
    # 4️⃣ Next-bar execution: shift signal so trades apply on t+1
    df['position'] = df['signal'].shift(1).fillna(0)
    
    # 5️⃣ Detect when trades occur (position changes)
    df['trade_change'] = df['position'].diff().abs().fillna(0)
    
    # 6️⃣ Simulate equity curve with costs/slippage
    eq = initial_equity
    equity_curve = []
    for pos, ret, change in zip(df['position'], df['return'], df['trade_change']):
        # 6a️⃣ Apply market return
        eq = eq * (1 + pos * ret)
        # 6b️⃣ Deduct transaction cost (round-trip) on each change
        cost = change * trade_cost * eq
        # 6c️⃣ Deduct slippage per new trade
        slip = change * slippage * eq
        eq -= (cost + slip)
        equity_curve.append(eq)
    
    # 7️⃣ Return a Series for easy downstream handling
    return pd.Series(equity_curve, index=df.index, name='equity')
