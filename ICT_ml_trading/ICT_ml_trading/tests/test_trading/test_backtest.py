# tests/test_trading/test_backtester.py
import pandas as pd
import numpy as np
import pytest
from src.trading.backtester import backtest_signals

@ pytest.fixture
def synthetic_price_and_signals():
    dates = pd.date_range('2025-01-01', periods=4, freq='H')
    # price jumps by +10%, then +10%, then -20%
    price = pd.DataFrame({'close': [100, 110, 121, 96.8]}, index=dates)
    # signals: always long
    signals_long = pd.DataFrame({'signal': [1, 1, 1, 1]}, index=dates)
    # signals: always short
    signals_short = pd.DataFrame({'signal': [-1, -1, -1, -1]}, index=dates)
    # signals: flat
    signals_flat = pd.DataFrame({'signal': [0, 0, 0, 0]}, index=dates)
    return price, signals_long, signals_short, signals_flat


def test_backtest_all_long(synthetic_price_and_signals):
    price, sig_long, _, _ = synthetic_price_and_signals
    eq = backtest_signals(sig_long, price, initial_equity=1000)
    # expected equity: start=1000
    # returns: [0, 0.10, 0.10, -0.20]
    ret = price['close'].pct_change().fillna(0)
    expected = 1000
    expected_curve = []
    for r in ret:
        expected = expected * (1 + 1 * r)
        expected_curve.append(expected)
    assert np.allclose(eq.values, expected_curve)


def test_backtest_all_short(synthetic_price_and_signals):
    price, _, sig_short, _ = synthetic_price_and_signals
    eq = backtest_signals(sig_short, price, initial_equity=1000)
    # For short: position = -1, equity multiplies by (1 - return)
    ret = price['close'].pct_change().fillna(0)
    expected = 1000
    expected_curve = []
    for r in ret:
        expected = expected * (1 - r)
        expected_curve.append(expected)
    assert np.allclose(eq.values, expected_curve)


def test_backtest_flat(synthetic_price_and_signals):
    price, _, _, sig_flat = synthetic_price_and_signals
    eq = backtest_signals(sig_flat, price, initial_equity=1000)
    # Always flat, equity remains constant
    assert np.all(eq.values == 1000)


def test_backtest_alignment_mismatch():
    # signals and price indices must align
    price = pd.DataFrame({'close': [100, 101]}, index=pd.date_range('2025-01-01', periods=2, freq='H'))
    signals = pd.DataFrame({'signal': [1]}, index=pd.date_range('2025-01-01', periods=1, freq='H'))
    with pytest.raises(KeyError):
        backtest_signals(signals, price, initial_equity=1000)
