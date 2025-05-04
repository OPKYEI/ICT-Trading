# src/utils/visualization.py

"""
Visualization utilities: plot equity curves, drawdowns, and metric summaries.
"""

import os
from typing import Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_equity_curve(
    equity: pd.Series,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot equity curve over time.

    Args:
        equity: pd.Series indexed by datetime.
        output_path: if given, save figure to this filepath (supports .png, .pdf).
        show: if True, call plt.show().
    """
    fig, ax = plt.subplots()
    ax.plot(equity.index, equity.values)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_drawdown(
    equity: pd.Series,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot drawdown over time.

    Args:
        equity: pd.Series indexed by datetime.
        output_path: if given, save figure.
        show: if True, call plt.show().
    """
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig, ax = plt.subplots()
    ax.plot(drawdown.index, drawdown.values)
    ax.set_title("Drawdown")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_metric_bar(
    metrics: Dict[str, float],
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot performance metrics as a bar chart.

    Args:
        metrics: dict of metric name → value.
        output_path: if given, save figure.
        show: if True, call plt.show().
    """
    names = list(metrics.keys())
    values = [metrics[k] for k in names]

    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_title("Performance Metrics")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)
