# src/trading/executor.py
"""
Execution logic: simulate a broker executing signals into trades.
"""
import pandas as pd
from typing import List, Dict

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
class OandaExecutor:
    def __init__(self, token, account):
        self.client = API(access_token=token, environment=config.OANDA_ENV)
        self.account = account

    def send_order(self, instrument, units, tp_price=None, sl_price=None):
        data = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                **({"takeProfitOnFill": {"price": tp_price}} if tp_price else {}),
                **({"stopLossOnFill": {"price": sl_price}} if sl_price else {})
            }
        }
        r = OrderCreate(accountID=self.account, data=data)
        return self.client.request(r)