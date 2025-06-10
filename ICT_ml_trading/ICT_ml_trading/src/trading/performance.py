# src/trading/performance.py
"""
Compute trading performance metrics from equity curves.
"""
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def compute_performance_metrics(
    equity: pd.Series,
    period_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate key performance metrics from an equity curve.

    Args:
        equity: pd.Series of equity values over time.
        period_per_year: Number of periods per year (e.g. 252 for daily bars).
        risk_free_rate: Annualized risk-free rate (e.g. 0.01 for 1%).

    Returns:
        Dictionary of metrics:
          - total_return
          - annual_return (CAGR)
          - annual_volatility
          - sharpe_ratio
          - sortino_ratio
          - max_drawdown
          - max_drawdown_duration (number of periods)
          - calmar_ratio
    """
    # Basic returns
    start_eq = equity.iloc[0]
    end_eq = equity.iloc[-1]
    total_periods = len(equity) - 1
    total_return = end_eq / start_eq - 1 if start_eq != 0 else np.nan

    # CAGR
    if total_periods > 0:
        annual_return = (end_eq / start_eq) ** (period_per_year / total_periods) - 1
    else:
        annual_return = np.nan

    # Period returns for volatility
    returns = equity.pct_change().dropna()
    # Annualized volatility
    annual_volatility = returns.std(ddof=0) * np.sqrt(period_per_year)

    # Sharpe Ratio
    excess_return = returns.mean() * period_per_year - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else np.nan

    # Sortino Ratio
    downside = returns[returns < 0]
    downside_vol = downside.std(ddof=0) * np.sqrt(period_per_year)
    sortino_ratio = excess_return / downside_vol if downside_vol > 0 else np.nan

    # Max Drawdown and Duration
    running_max = equity.cummax()
    drawdowns = (running_max - equity) / running_max
    max_drawdown = drawdowns.max()

    # Duration: longest consecutive drawdown periods
    dd_bool = drawdowns > 0
    max_dd_dur = 0
    current = 0
    for val in dd_bool:
        if val:
            current += 1
            max_dd_dur = max(max_dd_dur, current)
        else:
            current = 0

    # Calmar Ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else np.nan

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': float(max_dd_dur),
        'calmar_ratio': calmar_ratio
    }


def print_performance_metrics(
    metrics: Dict[str, float]
) -> None:
    """
    Print performance metrics to console.
    """
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def save_performance_metrics(
    metrics: Dict[str, float],
    output_path: str
) -> None:
    """
    Save performance metrics as CSV and JSON.

    Args:
        metrics: dict of metric names to values
        output_path: file path without extension
    """
    # DataFrame
    df = pd.DataFrame([metrics])
    df.to_csv(output_path + '.csv', index=False)
    # JSON
    with open(output_path + '.json', 'w') as f:
        json.dump(metrics, f, indent=4)
