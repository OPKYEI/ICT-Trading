# src/trading/backtester.py
"""
Backtesting framework: simulate equity curve from bar-by-bar signals.
"""
import pandas as pd

def backtest_signals(
    signals: pd.DataFrame,
    price: pd.DataFrame,
    initial_equity: float = 10000.0
) -> pd.Series:
    """
    Simple backtester that applies the previous bar's signal to next bar's return.

    Args:
        signals: DataFrame with 'signal' column (1=long, -1=short, 0=flat)
        price: DataFrame with 'close' prices
        initial_equity: starting capital

    Returns:
        equity_curve: pd.Series of equity over time
    """
    # Require exact alignment of indices
    if not signals.index.equals(price.index):
        raise KeyError("Signals and price index must align exactly")

    # Combined DataFrame for calculations
    df = pd.DataFrame(index=signals.index)
    df['signal'] = signals['signal']
    df['close'] = price['close']

    # Compute bar-to-bar returns
    df['return'] = df['close'].pct_change().fillna(0)

    # Use prior signal for position
    df['position'] = df['signal'].shift(1).fillna(0)

    # Generate equity curve
    eq = initial_equity
    equity_curve = []
    for pos, ret in zip(df['position'], df['return']):
        eq = eq * (1 + pos * ret)
        equity_curve.append(eq)

    return pd.Series(equity_curve, index=df.index, name='equity')
