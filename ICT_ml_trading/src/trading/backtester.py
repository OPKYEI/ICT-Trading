import pandas as pd
from src.utils.config import USE_TP_SL, TAKE_PROFIT_PIPS, STOP_LOSS_PIPS

def backtest_signals(
    signals,
    price: pd.DataFrame,
    initial_equity: float = 10_000.0,
    pip_size: float = 0.0001
) -> pd.Series:
    """
    Simulate and return the equity curve from timestamped signals.

    Args:
      signals: Series or DataFrame. If DataFrame, must have a 'signal' column.
      price:   DataFrame with 'high','low','close' columns.
      initial_equity: starting capital.
      pip_size: price units per pip (e.g. 0.0001 for EURUSD).
    Returns:
      pd.Series named 'equity'.
    """
    # ——— Normalize signals to a Series ———
    if isinstance(signals, pd.DataFrame):
        if 'signal' not in signals:
            raise ValueError("backtest_signals: DataFrame must have a 'signal' column")
        signals = signals['signal']
    # Now signals is guaranteed a Series

    # 1) Deduplicate & align
    if signals.index.duplicated().any():
        signals = signals[~signals.index.duplicated(keep='first')]
    if price.index.duplicated().any():
        price = price[~price.index.duplicated(keep='first')]

    idx = signals.index.intersection(price.index)
    df = price.loc[idx].copy()
    df['signal'] = signals.loc[idx]

    # 2) Compute bar returns and positions
    df['return']       = df['close'].pct_change().fillna(0)
    df['position']     = df['signal'].shift(1).fillna(0)
    df['trade_change'] = df['position'].diff().abs().fillna(0)

    # 3) Simulate equity
    eq = initial_equity
    equity_curve = []
    prev_close = None

    for pos, ret, change, close in zip(
        df['position'], df['return'], df['trade_change'], df['close']
    ):
        if prev_close is None:
            prev_close = close

        # base profit
        profit_pct = pos * ret

        # optional TP/SL clipping
        if USE_TP_SL and pos != 0:
            tp_pct = (TAKE_PROFIT_PIPS * pip_size) / prev_close
            sl_pct = (STOP_LOSS_PIPS   * pip_size) / prev_close
            profit_pct = min(profit_pct, tp_pct)
            profit_pct = max(profit_pct, -sl_pct)

        # apply profit
        eq *= (1 + profit_pct)

        # subtract slippage proportional to whether trade changed
        slip_pct = (pip_size / close) if close > 0 else 0
        eq -= change * slip_pct * eq

        equity_curve.append(eq)
        prev_close = close

    return pd.Series(equity_curve, index=df.index, name='equity')
