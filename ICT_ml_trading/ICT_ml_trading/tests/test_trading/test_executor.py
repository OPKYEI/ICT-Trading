# tests/test_trading/test_executor.py
import pandas as pd
import numpy as np
import pytest
from src.trading.executor import Executor

@pytest.fixture
def price_signals():
    dates = pd.date_range('2025-01-01', periods=5, freq='H')
    price = pd.DataFrame({'close': [100, 102, 101, 103, 104]}, index=dates)
    # signals: 0->1->1->0->-1
    signals = pd.DataFrame({'signal': [0, 1, 1, 0, -1]}, index=dates)
    return price, signals


def test_executor_basic(price_signals):
    price, signals = price_signals
    exec = Executor()
    trades = exec.execute(signals, price)
    # Expect two trades
    assert isinstance(trades, pd.DataFrame)
    assert len(trades) == 2
    dates = price.index
    # First trade: entry at idx 1, exit at idx 3
    first = trades.iloc[0]
    assert first['entry_time'] == dates[1]
    assert first['exit_time'] == dates[3]
    assert first['entry_price'] == 102
    assert first['exit_price'] == 103
    assert first['position'] == 1
    # Second trade: entry at idx 4, exit at idx 4
    second = trades.iloc[1]
    assert second['entry_time'] == dates[4]
    assert second['exit_time'] == dates[4]
    assert second['entry_price'] == 104
    assert second['exit_price'] == 104
    assert second['position'] == -1


def test_executor_alignment_mismatch():
    price = pd.DataFrame({'close': [100]}, index=pd.date_range('2025-01-01', periods=1, freq='H'))
    signals = pd.DataFrame({'signal': [1]}, index=pd.date_range('2025-01-02', periods=1, freq='H'))
    exec = Executor()
    with pytest.raises(KeyError):
        exec.execute(signals, price)
