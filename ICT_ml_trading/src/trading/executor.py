# src/trading/executor.py
"""
Execution logic: simulate a broker executing signals into trades.
"""
import pandas as pd
from typing import List, Dict
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate
import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate
from src.utils.config import (
    OANDA_API_TOKEN,
    OANDA_ACCOUNT_ID,
    SYMBOL,
    USE_TP_SL,
    TAKE_PROFIT_PIPS,
    STOP_LOSS_PIPS,
    PIP_SIZE_DICT,
    DEFAULT_PIP_SIZE,
)

class Executor:
    """
    Simulate trade execution based on bar-by-bar signals.
    """
    def __init__(self):
        pass

    def execute(
        self,
        signals: pd.DataFrame,
        price: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trade log from signals and price data.

        Rules:
          - Enter a position when signal changes from 0 to ±1 at that bar's close.
          - Exit (and possibly reverse) when signal changes to a different non-zero value or back to 0.
          - All fills at the bar's close price.
          - If position remains open at the end, close at last price.

        Returns:
          DataFrame with columns ['entry_time','exit_time','entry_price','exit_price','position']
        """
        # Ensure index alignment
        if not signals.index.equals(price.index):
            raise KeyError("Signals and price index must align exactly")

        trades: List[Dict] = []
        position = 0
        entry_time = None
        entry_price = None

        for t in signals.index:
            sig = int(signals.at[t, 'signal'])
            px = float(price.at[t, 'close'])
            # New entry
            if position == 0 and sig != 0:
                position = sig
                entry_time = t
                entry_price = px
            # Exit or reverse
            elif position != 0 and sig != position:
                exit_time = t
                exit_price = px
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position
                })
                # Start new if reversed
                if sig != 0:
                    position = sig
                    entry_time = t
                    entry_price = px
                else:
                    position = 0
                    entry_time = None
                    entry_price = None

        # Close any open position at end
        if position != 0:
            t = signals.index[-1]
            px = float(price.at[t, 'close'])
            trades.append({
                'entry_time': entry_time,
                'exit_time': t,
                'entry_price': entry_price,
                'exit_price': px,
                'position': position
            })

        return pd.DataFrame(trades)
# Add this improved OandaExecutor class to your src/trading/executor.py file
# Replace or update the existing OandaExecutor class

class OandaExecutor:
    """
    Live order executor using OANDA REST API.
    """

    def __init__(self):
        # Initialize OANDA REST client
        self.client = API(
            access_token=OANDA_API_TOKEN,
            environment="practice"  # switch to "live" if you change OANDA_ENV
        )
        self.account = OANDA_ACCOUNT_ID

        # Determine pip size once, from config
        clean_symbol = SYMBOL.replace("_", "")
        self.pip_size = PIP_SIZE_DICT.get(clean_symbol, DEFAULT_PIP_SIZE)

        print(f"✅ Initialized Oanda executor with account: {self.account} "
              f"and pip_size: {self.pip_size}")

    def place_order(self, signal, symbol=SYMBOL, units=1000, price=None):
        """
        Execute a market order on OANDA.
        signal: +1 buy, -1 sell, 0 skip
        units: base risk units
        price: float for logging/TP-SL calculation
        """
        if signal == 0:
            print("ℹ️ No trade signal (0), skipping order")
            return None

        # Scale base units by signal direction
        order_units = int(units * signal)

        # Use the precomputed pip_size
        pip_size = self.pip_size

        # Build order payload
        data = {
            "order": {
                "instrument": symbol,
                "units": str(order_units),
                "type": "MARKET",
                "timeInForce": "FOK",
            }
        }

        # Append TP/SL if configured
        if USE_TP_SL and price is not None:
            tp = price + signal * TAKE_PROFIT_PIPS * pip_size
            sl = price - signal * STOP_LOSS_PIPS * pip_size
            data["order"]["takeProfitOnFill"] = {"price": f"{tp:.5f}"}
            data["order"]["stopLossOnFill"] = {"price": f"{sl:.5f}"}
            tp_sl_info = f", TP={tp:.5f}, SL={sl:.5f}"
        else:
            tp_sl_info = ""

        # Log what we’re about to do
        ts = pd.Timestamp.now()
        side = "BUY" if signal > 0 else "SELL"
        price_info = f" @ {price:.5f}" if price is not None else ""
        print(f"🛎️ Sending {side} {abs(order_units)} units of {symbol}"
              f"{price_info} at {ts}{tp_sl_info}")

        try:
            req = OrderCreate(accountID=self.account, data=data)
            resp = self.client.request(req)
            tx_id = resp["orderCreateTransaction"]["id"]
            print(f"✅ Order executed successfully: {tx_id}")
            return resp
        except Exception as e:
            print(f"❌ Order execution failed: {e}")
            return None

    def send_order(self, timestamp, signal, price, pip_size=None):
        """
        Legacy wrapper; calls place_order.
        """
        return self.place_order(signal=signal, price=price)