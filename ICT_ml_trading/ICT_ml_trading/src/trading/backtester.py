import pandas as pd
from src.utils.config import USE_TP_SL, TAKE_PROFIT_PIPS, STOP_LOSS_PIPS
from src.trading.performance import compute_performance_metrics, save_performance_metrics

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


def enhanced_backtest_with_metrics(signals, price_data, symbol, initial_equity=10000, pip_size=0.0001):
    """
    Run backtest and display comprehensive performance metrics
    """
    # Run the backtest
    equity_curve = backtest_signals(signals, price_data, initial_equity, pip_size)
    
    # Determine periods per year based on timeframe
    if '60m' in symbol or '1h' in symbol:
        # 60-minute bars: ~8760 bars per year (24*365)
        periods_per_year = 8760
    elif '4h' in symbol:
        periods_per_year = 2190  # 24*365/4
    elif '1d' in symbol or 'daily' in symbol:
        periods_per_year = 252   # Trading days
    else:
        periods_per_year = 252   # Default fallback
    
    # Compute all metrics
    metrics = compute_performance_metrics(
        equity=equity_curve,
        period_per_year=periods_per_year,
        risk_free_rate=0.02  # 2% risk-free rate
    )
    
    # Enhanced display
    final_equity = equity_curve.iloc[-1]
    total_return_pct = (final_equity / initial_equity - 1) * 100
    
    print(f"\n🚀 ====== COMPREHENSIVE BACKTEST RESULTS for {symbol} ======")
    print(f"💰 Final Equity: ${final_equity:,.2f}")
    print(f"📈 Total Return: {total_return_pct:.2f}%")
    print(f"📊 Annual Return (CAGR): {metrics['annual_return']*100:.2f}%")
    print(f"📉 Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"⏱️  Max DD Duration: {metrics['max_drawdown_duration']:.0f} bars")
    print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"🎯 Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"🏆 Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"📊 Annual Volatility: {metrics['annual_volatility']*100:.2f}%")
    print(f"===============================================\n")
    
    # Save metrics to file (optional - create results directory if needed)
    try:
        import os
        os.makedirs("results", exist_ok=True)
        metrics_path = f"results/{symbol}_performance_metrics"
        save_performance_metrics(metrics, metrics_path)
        print(f"📁 Metrics saved to: {metrics_path}.csv and {metrics_path}.json")
    except Exception as e:
        print(f"⚠️ Could not save metrics to file: {e}")
    
    return equity_curve, metrics