# tests/test_trading/test_strategy.py

import numpy as np
import pandas as pd
import pytest
from src.trading.strategy import TradingStrategy
from src.ml_models.model_builder import ModelBuilder

@pytest.fixture
def simple_model():
    # Train a trivial model on a tiny dataset
    X = pd.DataFrame({'feat': [0, 1, 0, 1]}, index=pd.date_range('2025-01-01', periods=4, freq='H'))
    y = np.array([0, 1, 0, 1])
    model = ModelBuilder.build_logistic_regression(pca_components=None)
    model.fit(X, y)
    return model, X

def test_generate_signals_default_thresholds(simple_model):
    model, X = simple_model
    strat = TradingStrategy(model)
    signals = strat.generate_signals(X)
    assert isinstance(signals, pd.DataFrame)
    # All probabilities for X==1 should be >0.5 → signal=1
    assert all(signals.loc[X['feat'] == 1, 'signal'] == 1)
    # All probabilities for X==0 should be <0.5 → signal=-1
    assert all(signals.loc[X['feat'] == 0, 'signal'] == -1)

def test_generate_signals_custom_thresholds(simple_model):
    model, X = simple_model
    # Use wide band to produce some zeros
    strat = TradingStrategy(model, long_threshold=0.8, short_threshold=0.2)
    signals = strat.generate_signals(X)
    # For mid-range probabilities, expect some zeros
    assert 0 in signals['signal'].unique()
    assert set(signals['signal'].unique()).issubset({-1, 0, 1})
