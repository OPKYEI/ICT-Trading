# tests/test_trading/test_performance_metrics.py
import os
import json
import numpy as np
import pandas as pd
import pytest

from src.trading.performance import (
    compute_performance_metrics,
    print_performance_metrics,
    save_performance_metrics
)


def test_compute_metrics_monotonic_growth():
    # Equity grows 10% per period: [100,110,121]
    eq = pd.Series([100, 110, 121])
    metrics = compute_performance_metrics(eq, period_per_year=2, risk_free_rate=0.0)
    # total_return = 1.21 - 1 = 0.21
    assert np.isclose(metrics['total_return'], 0.21)
    # annual_return with 2 periods per year equals total_return
    assert np.isclose(metrics['annual_return'], 0.21)
    # No drawdown on monotonic growth
    assert np.isclose(metrics['max_drawdown'], 0.0)
    assert metrics['max_drawdown_duration'] == 0.0
    # Volatility is zero because returns are constant -> nan Sharpe/Sortino
    assert np.isnan(metrics['annual_volatility']) or np.isclose(metrics['annual_volatility'], 0)
    assert np.isnan(metrics['sharpe_ratio'])
    assert np.isnan(metrics['sortino_ratio'])
    # Calmar ratio undefined for zero drawdown
    assert np.isnan(metrics['calmar_ratio'])


def test_compute_metrics_with_drawdown():
    # Equity peaks then drops: [100, 120, 80, 140]
    eq = pd.Series([100, 120, 80, 140])
    metrics = compute_performance_metrics(eq, period_per_year=4, risk_free_rate=0.0)
    # total_return = 140/100 -1 = 0.4
    assert np.isclose(metrics['total_return'], 0.4)
    # max drawdown: from 120 -> 80 = 40/120 = 0.3333
    assert np.isclose(metrics['max_drawdown'], (120 - 80) / 120)
    # duration = 1 period drawdown
    assert np.isclose(metrics['max_drawdown_duration'], 1.0)
    # positive volatility and ratios
    assert metrics['annual_volatility'] > 0
    # save/print will be tested separately


def test_print_and_save_metrics(tmp_path, capsys):
    metrics = {'a': 1.2345, 'b': 0.5}
    # Test print
    print_performance_metrics(metrics)
    captured = capsys.readouterr()
    assert 'a: 1.2345' in captured.out
    assert 'b: 0.5000' in captured.out

    # Test save
    out = tmp_path / 'perf'
    save_performance_metrics(metrics, str(out))
    # Check files
    csv_path = str(out) + '.csv'
    json_path = str(out) + '.json'
    assert os.path.exists(csv_path)
    assert os.path.exists(json_path)

    df = pd.read_csv(csv_path)
    assert np.isclose(df['a'].iloc[0], 1.2345)
    data = json.load(open(json_path))
    assert data['b'] == 0.5
