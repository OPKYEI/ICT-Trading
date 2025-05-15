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
    SYMBOLS,
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
            # Ensure index alignment (allow reindexing to match)
        if not signals.index.equals(price.index):
            # try to reindex price to the signals index
            price = price.reindex(signals.index)
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
            environment="practice"  # or use OANDA_ENV if you import it
        )
        self.account = OANDA_ACCOUNT_ID
        print(f"✅ Initialized Oanda executor with account: {self.account}")

    def place_order(self, signal: int, symbol: str, units: int = 1000, price: float = None):
        """
        Execute a market order on OANDA.
        - signal: +1 buy, -1 sell, 0 skip
        - symbol: e.g. "EUR_USD"
        - units: base unit size
        - price: float for calculating TP/SL
        """

        if signal == 0:
            print("ℹ️ No trade signal (0), skipping order")
            return None

        # Clean up the symbol for OANDA (remove underscore)
        instrument = symbol

        # Determine pip size *for this instrument*
        pip_size = PIP_SIZE_DICT.get(instrument, DEFAULT_PIP_SIZE)

        # Compute absolute units
        order_units = int(units * signal)

        # Build order payload
        data = {
            "order": {
                "instrument": instrument,
                "units": str(order_units),
                "type": "MARKET",
                "timeInForce": "FOK",
            }
        }

        # Add TP/SL if enabled
        tp_sl_info = ""
        if USE_TP_SL and price is not None:
            tp_price = price + signal * TAKE_PROFIT_PIPS * pip_size
            sl_price = price - signal * STOP_LOSS_PIPS * pip_size
            data["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.5f}"}
            data["order"]["stopLossOnFill"]   = {"price": f"{sl_price:.5f}"}
            tp_sl_info = f", TP={tp_price:.5f}, SL={sl_price:.5f}"

        # Log the outgoing order
        ts   = pd.Timestamp.now()
        side = "BUY" if signal > 0 else "SELL"
        price_info = f" @ {price:.5f}" if price is not None else ""
        print(f"🛎️ Sending {side} {abs(order_units)} units of {instrument}"
              f"{price_info} at {ts}{tp_sl_info}")

        # Send the request
        try:
            req  = OrderCreate(accountID=self.account, data=data)
            resp = self.client.request(req)
            tx_id = resp["orderCreateTransaction"]["id"]
            print(f"✅ Order executed successfully: {tx_id}")
            return resp
        except Exception as e:
            print(f"❌ Order execution failed: {e}")
            return None

    def send_order(self, timestamp, signal, price, pip_size=None):
        """
        Legacy wrapper pointing to place_order
        """
        return self.place_order(signal=signal, price=price)
import MetaTrader5 as mt5

class FTMOExecutor:
    def __init__(self, terminal: str, login: int, password: str, server: str):
        # Initialize the MT5 terminal connection
        if not mt5.initialize(path=terminal, login=login, password=password, server=server):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    def place_order(self, signal: int, symbol: str, units: int, price: float = None) -> dict:
        """
        Send a market order via MT5 using FTMO demo account.
        
        signal: +1 for buy, -1 for sell
        symbol: e.g. "EUR_USD" (will be converted to "EURUSD")
        units: in base‐currency units (e.g. 1000 → 0.01 lots)
        price: ignored for market orders
        """
        # MT5 uses no underscore, and 1 lot = 100,000 units
        MT5_SYMBOL_MAP = {
            "EUR_USD":  "EURUSD",
            "XAU_USD":  "XAUUSD",
            "US30_USD": "US30.cash",    # adjust to your terminal’s exact name
            "NAS100_USD":"NAS100.cash"  # adjust as needed
        }
        mt5_sym = MT5_SYMBOL_MAP.get(symbol, symbol.replace("_",""))
        # ensure the symbol is available in Market Watch
        if not mt5.symbol_select(mt5_sym, True):
            raise RuntimeError(f"MT5 symbol_select failed for {mt5_sym}")

        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            raise RuntimeError(f"MT5 no tick data for {mt5_sym}")

        # build the order request
        order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
        lot_size   = units / 100_000.0
        price_to_use = tick.ask if signal > 0 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_sym,
            "volume": lot_size,
            "type": order_type,
            "price": price_to_use,
            "deviation": 10,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "magic": 234000,
            "comment": "FTMO demo trade"
        }

        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"MT5 order_send failed: {mt5.last_error()}")
        # Convert the result (a namedtuple) to dict if possible
        return result._asdict() if hasattr(result, "_asdict") else dict(result)

    def __del__(self):
        try:
            mt5.shutdown()
        except:
            pass
