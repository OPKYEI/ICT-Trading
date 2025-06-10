# tests/test_trading/test_risk_manager.py
import pytest
import numpy as np
from src.trading.risk_manager import RiskManager

@ pytest.fixture
def risk_manager():
    # Account equity of 10000, risk 1% per trade
    return RiskManager(account_equity=10000.0, risk_per_trade=0.01)


def test_calculate_position_size_basic(risk_manager):
    entry = 1.20
    stop = 1.10
    # risk_amount = 10000 * 0.01 = 100
    # risk_per_unit = 0.10
    # size = 100 / 0.10 = 1000
    size = risk_manager.calculate_position_size(entry, stop)
    assert np.isclose(size, 1000.0)


def test_calculate_position_size_zero_diff(risk_manager):
    with pytest.raises(ValueError):
        risk_manager.calculate_position_size(1.0, 1.0)


def test_calculate_stop_loss_long():
    entry = 1.20
    atr = 0.02
    stop = RiskManager.calculate_stop_loss(entry_price=entry, atr=atr, direction='long', multiplier=2.0)
    assert np.isclose(stop, 1.20 - 0.02 * 2.0)


def test_calculate_stop_loss_short():
    entry = 1.20
    atr = 0.02
    stop = RiskManager.calculate_stop_loss(entry_price=entry, atr=atr, direction='short', multiplier=2.0)
    assert np.isclose(stop, 1.20 + 0.02 * 2.0)


def test_calculate_stop_loss_invalid_direction():
    with pytest.raises(ValueError):
        RiskManager.calculate_stop_loss(1.0, 0.01, direction='up', multiplier=1.0)


def test_calculate_stop_loss_negative_multiplier():
    with pytest.raises(ValueError):
        RiskManager.calculate_stop_loss(1.0, 0.01, direction='long', multiplier=-1.0)
