# src/trading/executor.py
"""
Execution logic: simulate a broker executing signals into trades.
Updated with SIMPLE unit standardization: 10,000 units = 0.1 lots always.
"""
import time
import pandas as pd
from typing import List, Dict
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate
import MetaTrader5 as mt5

from src.utils.config import (
    OANDA_API_TOKEN,
    OANDA_ACCOUNT_ID,
    OANDA_ENV,
    SYMBOLS,
    USE_TP_SL,
    TAKE_PROFIT_PIPS,
    STOP_LOSS_PIPS,
    PIP_SIZE_DICT,
    DEFAULT_PIP_SIZE,
)

def simple_units_to_lots(units: float) -> float:
    """
    SIMPLE STANDARDIZATION: Convert units to lots using fixed formula
    
    Rules:
    - 100,000 units = 1.0 lot
    - 10,000 units = 0.1 lot  
    - 1,000 units = 0.01 lot
    
    Args:
        units: Number of units from trading system
        
    Returns:
        float: Lot size
    """
    # Simple conversion: divide by 100,000
    lot_size = units / 100_000
    
    # Ensure minimum lot size (most brokers require 0.01 minimum)
    lot_size = max(0.01, lot_size)
    
    # Round to 2 decimal places (standard lot precision)
    lot_size = round(lot_size, 2)
    
    return lot_size


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


class OandaExecutor:
    """
    Live order executor using OANDA REST API.
    """

    def __init__(self):
        # Initialize OANDA REST client
        self.client = API(access_token=OANDA_API_TOKEN, environment=OANDA_ENV)
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


class FTMOExecutor:
    def __init__(self, terminal: str, login: int, password: str, server: str):
        self.terminal = terminal
        self.login = login
        self.password = password
        self.server = server
        
        # Corrected FTMO symbol mapping based on actual FTMO symbols
        self.SYMBOL_MAP = {
            "EUR_USD": "EURUSD",
            "GBP_USD": "GBPUSD", 
            "USD_JPY": "USDJPY",
            "AUD_USD": "AUDUSD",
            "USD_CAD": "USDCAD",
            "XAU_USD": "XAUUSD",  # Gold is usually without .cash on FTMO
            "XAG_USD": "XAGUSD",  # Silver
            "US30_USD": "US30.cash",     # Dow Jones 30
            "NAS100_USD": "US100.cash",  # Nasdaq 100
            # Alternative mappings in case the above don't work
            "USA30": "US30.cash",
            "USATECH": "US100.cash",
        }
        
        # Cache for contract specifications
        self._contract_specs = {}
        
        print(f"✅ FTMO executor initialized")
        
    def _ensure_connected(self):
        """Ensure we're connected to the right MT5 terminal"""
        account_info = mt5.account_info()
        if account_info and account_info.login == self.login:
            return True
            
        try:
            mt5.shutdown()
            time.sleep(0.2)
        except:
            pass
            
        if not mt5.initialize(
            path=self.terminal,
            login=self.login,
            password=self.password,
            server=self.server,
            timeout=30000
        ):
            raise RuntimeError(f"Failed to connect to FTMO: {mt5.last_error()}")
            
        return True
    
    def _map_symbol(self, symbol):
        """Map trading symbol to FTMO MT5 symbol"""
        return self.SYMBOL_MAP.get(symbol, symbol.replace("_", ""))
    
    def get_contract_specification(self, symbol):
        """Get contract specification for a symbol"""
        mt5_symbol = self._map_symbol(symbol)
        
        # Return cached if available
        if mt5_symbol in self._contract_specs:
            return self._contract_specs[mt5_symbol]
        
        self._ensure_connected()
        
        # Try to select the symbol first
        if not mt5.symbol_select(mt5_symbol, True):
            print(f"⚠️ FTMO: Cannot select symbol {mt5_symbol}")
            return None
        
        symbol_info = mt5.symbol_info(mt5_symbol)
        if symbol_info is None:
            print(f"⚠️ FTMO: No symbol info for {mt5_symbol}")
            return None
        
        spec = {
            'contract_size': symbol_info.trade_contract_size,
            'tick_size': symbol_info.point,
            'tick_value': symbol_info.trade_tick_value,
            'min_lot': symbol_info.volume_min,
            'max_lot': symbol_info.volume_max,
            'lot_step': symbol_info.volume_step,
            'symbol_name': mt5_symbol
        }
        
        # Cache the result
        self._contract_specs[mt5_symbol] = spec
        
        print(f"📊 FTMO: {mt5_symbol} contract_size={spec['contract_size']}, "
              f"min_lot={spec['min_lot']}, lot_step={spec['lot_step']}")
        
        return spec
    
    def calculate_lot_size(self, symbol, desired_units):
        """
        SIMPLE STANDARDIZATION: Convert units to lots ignoring contract specs
        
        10,000 units = 0.1 lot (always, regardless of symbol)
        100,000 units = 1.0 lot (always, regardless of symbol)
        1,000 units = 0.01 lot (always, regardless of symbol)
        """
        # Get broker constraints (min_lot, lot_step) but ignore contract_size
        spec = self.get_contract_specification(symbol)
        if not spec:
            # Fallback values if we can't get specs
            min_lot = 0.01
            lot_step = 0.01
        else:
            min_lot = spec['min_lot']
            lot_step = spec['lot_step']
        
        # SIMPLE CONVERSION: units ÷ 100,000 = lots
        lot_size = simple_units_to_lots(desired_units)
        
        # Apply broker constraints
        lot_size = max(min_lot, lot_size)  # Respect minimum
        lot_size = round(lot_size / lot_step) * lot_step  # Respect step size
        lot_size = round(lot_size, 2)  # Clean up decimals
        
        print(f"💰 FTMO: {symbol} -> {desired_units} units = {lot_size} lots (standardized)")
        
        return lot_size
    
    def _debug_available_symbols(self, search_term):
        """Debug helper to find available symbols"""
        try:
            self._ensure_connected()
            symbols = mt5.symbols_get()
            if symbols:
                matches = [s.name for s in symbols if search_term.upper() in s.name.upper()]
                print(f"🔍 FTMO symbols containing '{search_term}': {matches[:10]}")
                return matches
        except Exception as e:
            print(f"❌ Error searching symbols: {e}")
        return []
        
    def place_order(self, signal: int, symbol: str, units: int, price: float = None) -> dict:
        """
        Send a market order via MT5 using FTMO demo account.
        Now with SIMPLE standardized position sizing!
        """
        try:
            self._ensure_connected()
            
            # Map the symbol
            mt5_symbol = self._map_symbol(symbol)
            print(f"🔄 FTMO: Mapping {symbol} → {mt5_symbol}")
            
            # Try to select the symbol
            if not mt5.symbol_select(mt5_symbol, True):
                # Symbol not found, try alternatives
                print(f"⚠️ FTMO: {mt5_symbol} not available, searching alternatives...")
                
                # Search for similar symbols
                if "US30" in symbol or "USA30" in symbol:
                    alternatives = self._debug_available_symbols("US30")
                    if alternatives:
                        mt5_symbol = alternatives[0]
                        print(f"🔄 FTMO: Using alternative symbol: {mt5_symbol}")
                elif "NAS100" in symbol or "USATECH" in symbol:
                    alternatives = self._debug_available_symbols("US100")
                    if not alternatives:
                        alternatives = self._debug_available_symbols("NAS")
                    if alternatives:
                        mt5_symbol = alternatives[0]
                        print(f"🔄 FTMO: Using alternative symbol: {mt5_symbol}")
                
                # Try again with the alternative
                if not mt5.symbol_select(mt5_symbol, True):
                    raise RuntimeError(f"FTMO: Cannot select symbol {mt5_symbol}")

            # Get tick data
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                raise RuntimeError(f"FTMO: No tick data for {mt5_symbol}")

            # SIMPLE STANDARDIZED POSITION SIZING!
            lot_size = self.calculate_lot_size(symbol, units)

            order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
            price_to_use = tick.ask if signal > 0 else tick.bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price_to_use,
                "deviation": 20,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "magic": 234000,
                "comment": f"FTMO {symbol}"
            }

            side = "BUY" if signal > 0 else "SELL"
            print(f"🛎️ FTMO: {side} {lot_size} lots of {mt5_symbol} @ {price_to_use}")

            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError(f"FTMO: order_send failed: {mt5.last_error()}")
                
            result_dict = result._asdict() if hasattr(result, "_asdict") else dict(result)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ FTMO: Order {result.order} executed successfully")
                
                # Get position ticket
                time.sleep(0.1)
                positions = mt5.positions_get(symbol=mt5_symbol)
                if positions:
                    our_positions = [p for p in positions if p.magic == 234000]
                    if our_positions:
                        result_dict['position_ticket'] = our_positions[-1].ticket
            else:
                print(f"❌ FTMO: Order failed - {result.comment} (retcode: {result.retcode})")
            
            return result_dict
            
        except Exception as e:
            print(f"❌ FTMO order failed for {symbol}: {e}")
            return {"retcode": -1, "comment": str(e)}

    def get_open_positions(self, symbol: str = None) -> list:
        """Get open positions, optionally filtered by symbol"""
        self._ensure_connected()
        
        if symbol:
            mt5_symbol = self._map_symbol(symbol)
            positions = mt5.positions_get(symbol=mt5_symbol)
        else:
            positions = mt5.positions_get()
            
        if positions is None:
            return []
        
        return [pos._asdict() if hasattr(pos, "_asdict") else dict(pos) for pos in positions]

    def close_position(self, ticket: int) -> dict:
        """Close a specific position by ticket ID"""
        self._ensure_connected()
        
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise RuntimeError(f"Position {ticket} not found")
        
        position = positions[0]
        mt5_symbol = position.symbol
        
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {mt5_symbol}")
        
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY  
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": f"Close position {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"Failed to close position {ticket}: {mt5.last_error()}")
        
        return result._asdict() if hasattr(result, "_asdict") else dict(result)

    def close_positions_by_symbol_and_magic(self, symbol: str, magic: int) -> list:
        """Close all positions for a symbol with specific magic number"""
        self._ensure_connected()
        
        mt5_symbol = self._map_symbol(symbol)
        positions = mt5.positions_get(symbol=mt5_symbol)
        
        if not positions:
            return []
        
        closed_positions = []
        for position in positions:
            if position.magic == magic:
                try:
                    result = self.close_position(position.ticket)
                    closed_positions.append(result)
                    print(f"✅ FTMO: Closed position {position.ticket}")
                except Exception as e:
                    print(f"❌ FTMO: Failed to close position {position.ticket}: {e}")
        
        return closed_positions


class PepperstoneExecutor:
    def __init__(self, terminal: str, login: int, password: str, server: str):
        self.terminal = terminal
        self.login = login
        self.password = password
        self.server = server
        
        # Pepperstone symbol mapping
        self.SYMBOL_MAP = {
            "EUR_USD": "EURUSD",
            "GBP_USD": "GBPUSD",
            "USD_JPY": "USDJPY", 
            "AUD_USD": "AUDUSD",
            "USD_CAD": "USDCAD",
            "XAU_USD": "XAUUSD",
            "XAG_USD": "XAGUSD",
            "US30_USD": "US30",      # Check actual Pepperstone symbol
            "NAS100_USD": "NAS100",  # Check actual Pepperstone symbol
        }
        
        # Cache for contract specifications
        self._contract_specs = {}
        
        print(f"✅ Pepperstone executor initialized")

    def _ensure_connected(self):
        """Ensure we're connected to the right MT5 terminal"""
        account_info = mt5.account_info()
        if account_info and account_info.login == self.login:
            return True
            
        try:
            mt5.shutdown()
            time.sleep(0.2)
        except:
            pass
            
        if not mt5.initialize(
            path=self.terminal,
            login=self.login,
            password=self.password,
            server=self.server,
            timeout=30000
        ):
            raise RuntimeError(f"Failed to connect to Pepperstone: {mt5.last_error()}")
            
        account_info = mt5.account_info()
        if not account_info or account_info.login != self.login:
            raise RuntimeError(f"Connected to wrong account: {account_info.login if account_info else 'None'}")
            
        return True
    
    def _map_symbol(self, symbol):
        """Map trading symbol to Pepperstone MT5 symbol"""
        return self.SYMBOL_MAP.get(symbol, symbol.replace("_", ""))
    
    def get_contract_specification(self, symbol):
        """Get contract specification for a symbol"""
        mt5_symbol = self._map_symbol(symbol)
        
        # Return cached if available
        if mt5_symbol in self._contract_specs:
            return self._contract_specs[mt5_symbol]
        
        self._ensure_connected()
        
        # Try to select the symbol first
        if not mt5.symbol_select(mt5_symbol, True):
            print(f"⚠️ Pepperstone: Cannot select symbol {mt5_symbol}")
            return None
        
        symbol_info = mt5.symbol_info(mt5_symbol)
        if symbol_info is None:
            print(f"⚠️ Pepperstone: No symbol info for {mt5_symbol}")
            return None
        
        spec = {
            'contract_size': symbol_info.trade_contract_size,
            'tick_size': symbol_info.point,
            'tick_value': symbol_info.trade_tick_value,
            'min_lot': symbol_info.volume_min,
            'max_lot': symbol_info.volume_max,
            'lot_step': symbol_info.volume_step,
            'symbol_name': mt5_symbol
        }
        
        # Cache the result
        self._contract_specs[mt5_symbol] = spec
        
        print(f"📊 Pepperstone: {mt5_symbol} contract_size={spec['contract_size']}, "
              f"min_lot={spec['min_lot']}, lot_step={spec['lot_step']}")
        
        return spec
    
    def calculate_lot_size(self, symbol, desired_units):
        """
        SIMPLE STANDARDIZATION: Convert units to lots ignoring contract specs
        
        10,000 units = 0.1 lot (always, regardless of symbol)
        100,000 units = 1.0 lot (always, regardless of symbol)
        1,000 units = 0.01 lot (always, regardless of symbol)
        """
        # Get broker constraints (min_lot, lot_step) but ignore contract_size
        spec = self.get_contract_specification(symbol)
        if not spec:
            # Fallback values if we can't get specs
            min_lot = 0.01
            lot_step = 0.01
        else:
            min_lot = spec['min_lot']
            lot_step = spec['lot_step']
        
        # SIMPLE CONVERSION: units ÷ 100,000 = lots
        lot_size = simple_units_to_lots(desired_units)
        
        # Apply broker constraints
        lot_size = max(min_lot, lot_size)  # Respect minimum
        lot_size = round(lot_size / lot_step) * lot_step  # Respect step size
        lot_size = round(lot_size, 2)  # Clean up decimals
        
        print(f"💰 Pepperstone: {symbol} -> {desired_units} units = {lot_size} lots (standardized)")
        
        return lot_size

    def place_order(self, signal: int, symbol: str, units: int, price: float = None) -> dict:
        """
        Execute order with SIMPLE standardized position sizing
        """
        try:
            self._ensure_connected()
            
            mt5_symbol = self._map_symbol(symbol)
            print(f"🔄 Pepperstone: Mapping {symbol} → {mt5_symbol}")
            
            if not mt5.symbol_select(mt5_symbol, True):
                # Try to find available symbols
                available = [s.name for s in mt5.symbols_get() if symbol[:3] in s.name]
                raise RuntimeError(f"Pepperstone: Cannot select {mt5_symbol}. Similar symbols: {available[:5]}")

            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                raise RuntimeError(f"Pepperstone: No tick data for {mt5_symbol}")

            # SIMPLE STANDARDIZED POSITION SIZING!
            lot_size = self.calculate_lot_size(symbol, units)
            
            order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
            price_to_use = tick.ask if signal > 0 else tick.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price_to_use,
                "deviation": 20,
                "magic": 234001,
                "comment": f"Pepperstone {symbol}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            side = "BUY" if signal > 0 else "SELL"
            print(f"🛎️ Pepperstone: {side} {lot_size} lots of {mt5_symbol} @ {price_to_use}")
            
            result = mt5.order_send(request)
            
            if result is None:
                raise RuntimeError(f"Pepperstone: order_send returned None")
            
            result_dict = result._asdict() if hasattr(result, "_asdict") else dict(result)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Pepperstone: Order {result.order} executed successfully")
                
                time.sleep(0.1)
                positions = mt5.positions_get(symbol=mt5_symbol)
                if positions:
                    our_positions = [p for p in positions if p.magic == 234001]
                    if our_positions:
                        result_dict['position_ticket'] = our_positions[-1].ticket
            else:
                print(f"❌ Pepperstone: Order failed - {result.comment} (retcode: {result.retcode})")
                
            return result_dict
            
        except Exception as e:
            print(f"❌ Pepperstone order failed for {symbol}: {e}")
            return {"retcode": -1, "comment": str(e)}

    def get_open_positions(self, symbol: str = None) -> list:
        """Get open positions, optionally filtered by symbol"""
        self._ensure_connected()
        
        if symbol:
            mt5_symbol = self._map_symbol(symbol)
            positions = mt5.positions_get(symbol=mt5_symbol)
        else:
            positions = mt5.positions_get()
            
        if positions is None:
            return []
        
        return [pos._asdict() if hasattr(pos, "_asdict") else dict(pos) for pos in positions]

    def close_position(self, ticket: int) -> dict:
        """Close a specific position by ticket ID"""
        self._ensure_connected()
        
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise RuntimeError(f"Position {ticket} not found")
        
        position = positions[0]
        mt5_symbol = position.symbol
        
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {mt5_symbol}")
        
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY  
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": f"Close position {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"Failed to close position {ticket}: {mt5.last_error()}")
        
        return result._asdict() if hasattr(result, "_asdict") else dict(result)

    def close_positions_by_symbol_and_magic(self, symbol: str, magic: int) -> list:
        """Close all positions for a symbol with specific magic number"""
        self._ensure_connected()
        
        mt5_symbol = self._map_symbol(symbol)
        positions = mt5.positions_get(symbol=mt5_symbol)
        
        if not positions:
            return []
        
        closed_positions = []
        for position in positions:
            if position.magic == magic:
                try:
                    result = self.close_position(position.ticket)
                    closed_positions.append(result)
                    print(f"✅ Pepperstone: Closed position {position.ticket}")
                except Exception as e:
                    print(f"❌ Pepperstone: Failed to close position {position.ticket}: {e}")
        
        return closed_positions

    def __del__(self):
        try:
            mt5.shutdown()
        except:
            pass