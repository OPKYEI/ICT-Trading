from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from src.utils.config import OANDA_API_TOKEN, OANDA_ACCOUNT_ID, OANDA_ENV

api = API(access_token=OANDA_API_TOKEN, environment=OANDA_ENV)
params = {"granularity": "M1", "count": 1}
r = InstrumentsCandles(instrument="EUR_USD", params=params)

try:
    data = api.request(r)
    print("✅ OK – got", data["candles"][0]["mid"])
except Exception as e:
    print("❌", e)
