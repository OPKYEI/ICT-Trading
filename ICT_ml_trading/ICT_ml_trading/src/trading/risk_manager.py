# src/trading/risk_manager.py
"""
Risk management: position sizing and stop-loss calculation.
"""
from dataclasses import dataclass

@dataclass
class RiskManager:
    account_equity: float      # total equity in account
    risk_per_trade: float      # fraction of equity to risk per trade, e.g., 0.01 for 1%

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Calculate position size (units) such that risk does not exceed risk_per_trade of equity.

        Args:
            entry_price: price at which trade is entered
            stop_price: price at which stop-loss is set

        Returns:
            Number of units/contracts to trade
        """
        risk_amount = self.account_equity * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            raise ValueError("Entry and stop prices must differ to calculate risk per unit.")
        return risk_amount / risk_per_unit

    @staticmethod
    def calculate_stop_loss(
        entry_price: float,
        atr: float,
        direction: str,
        multiplier: float = 1.5
    ) -> float:
        """
        Compute stop-loss level based on ATR.

        Args:
            entry_price: entry price
            atr: average true range value
            direction: 'long' or 'short'
            multiplier: ATR multiple to set stop distance

        Returns:
            Stop-loss price
        """
        if multiplier < 0:
            raise ValueError("Multiplier must be non-negative.")
        if direction == 'long':
            return entry_price - atr * multiplier
        elif direction == 'short':
            return entry_price + atr * multiplier
        else:
            raise ValueError("Direction must be 'long' or 'short'.")
