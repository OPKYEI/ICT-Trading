import logging
from src.utils.config import BROKER_NAME, OANDA_API_TOKEN, OANDA_ACCOUNT_ID, FXCM_API_TOKEN

class DemoBrokerClient:
    """
    Simple wrapper for a demo-account API client.
    Falls back to OANDA or FXCM if FTMO has no public API.
    """
    def __init__(self):
        if BROKER_NAME.upper() == "OANDA":
            from oandapyV20 import API
            self.client = API(access_token=OANDA_API_TOKEN, environment="practice")
        elif BROKER_NAME.upper() == "FXCM":
            import fxcmpy
            self.client = fxcmpy.fxcmpy(access_token=FXCM_API_TOKEN, log_level="error", server="demo")
        else:
            # FTMO has no direct API: force user to switch
            raise RuntimeError(
                "FTMO does not provide an automated API. "
                "Please change BROKER_NAME to 'OANDA' or 'FXCM' in config.py."
            )

    def get_latest_price(self, instrument: str):
        """
        Return the latest bid/ask or mid price for `instrument`.
        """
        if BROKER_NAME.upper() == "OANDA":
            from oandapyV20.endpoints.pricing import PricingInfo
            r = PricingInfo(accountID=OANDA_ACCOUNT_ID, params={"instruments": instrument})
            resp = self.client.request(r)
            quotes = resp.get("prices", [])[0]
            return (float(quotes["closeoutBid"]) + float(quotes["closeoutAsk"])) / 2
        elif BROKER_NAME.upper() == "FXCM":
            data = self.client.get_last_price(instrument)
            return (data["Bid"] + data["Ask"]) / 2

    def place_order(self, instrument: str, units: int, tp: float = None, sl: float = None):
        """
        Place a market order with optional TP/SL.
        """
        if BROKER_NAME.upper() == "OANDA":
            from oandapyV20.endpoints.orders import OrderCreate
            order = {
                "order": {
                    "instrument": instrument,
                    "units": str(units),
                    "type": "MARKET",
                    **({"takeProfitOnFill": {"price": str(tp)}} if tp else {}),
                    **({"stopLossOnFill": {"price": str(sl)}} if sl else {}),
                }
            }
            resp = self.client.request(OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order))
            return resp
        elif BROKER_NAME.upper() == "FXCM":
            return self.client.open_trade(
                symbol=instrument,
                is_buy=(units > 0),
                amount=abs(units),
                order_type="AtMarket",
                time_in_force="GTC",
                limit=tp,
                stop=sl,
            )
