# tests/test_utils/test_visualization.py

import os
import pandas as pd
import numpy as np
import pytest

from src.utils.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_metric_bar
)

@pytest.fixture
def synthetic_equity(tmp_path):
    dates = pd.date_range('2025-01-01', periods=10, freq='D')
    eq = pd.Series(1000 * (1 + np.linspace(0, 0.1, 10)), index=dates)
    return eq, tmp_path / "eq.png", tmp_path / "dd.png", tmp_path / "metrics.png"

def test_plot_equity_curve_creates_file_and_runs(synthetic_equity):
    eq, eq_path, _, _ = synthetic_equity
    # Should run without error and save file
    plot_equity_curve(eq, output_path=str(eq_path), show=False)
    assert os.path.exists(str(eq_path))

def test_plot_drawdown_creates_file_and_runs(synthetic_equity):
    eq, _, dd_path, _ = synthetic_equity
    plot_drawdown(eq, output_path=str(dd_path), show=False)
    assert os.path.exists(str(dd_path))

def test_plot_metric_bar_creates_file_and_runs(synthetic_equity):
    _, _, _, metrics_path = synthetic_equity
    metrics = {'sharpe_ratio': 1.23, 'max_drawdown': -0.05}
    plot_metric_bar(metrics, output_path=str(metrics_path), show=False)
    assert os.path.exists(str(metrics_path))

def test_plots_without_saving(synthetic_equity):
    eq, _, _, _ = synthetic_equity
    # Should not raise even if no output_path
    plot_equity_curve(eq, show=False)
    plot_drawdown(eq, show=False)
    plot_metric_bar({'a': 0.1, 'b': 0.2}, show=False)
