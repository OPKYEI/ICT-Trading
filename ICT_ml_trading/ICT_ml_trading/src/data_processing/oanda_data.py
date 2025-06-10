# src/data_processing/oanda_data.py

import pandas as pd
import logging
from datetime import timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

from src.utils.config import OANDA_API_TOKEN, OANDA_ACCOUNT_ID, OANDA_ENV

logger = logging.getLogger("oanda_data")
logger.setLevel(logging.INFO)


class OandaDataFetcher:
    """
    Fetch OHLCV data directly from OANDA REST API for live or practice accounts.
    """

    def __init__(self):
        """
        Initialize the Oanda API client using config settings.
        """
        # environment is either "practice" or "live"
        self.client = API(
            access_token=OANDA_API_TOKEN,
            environment=OANDA_ENV
        )
        self.account_id = OANDA_ACCOUNT_ID
        logger.info(f"Initialized OANDA data fetcher (env={OANDA_ENV}, account={self.account_id})")

    def fetch_ohlc(self, instrument: str, granularity: str = "H1", count: int = 120) -> pd.DataFrame:
        """
        Fetch the last `count` candles at given granularity.
        Returns a DataFrame indexed by timestamp with ['open','high','low','close','volume'].
        """
        try:
            params = {
                "granularity": granularity,
                "count": count,
                "price": "M"   # mid prices
            }
            req = InstrumentsCandles(instrument=instrument, params=params)
            resp = self.client.request(req)

            candles = resp.get("candles", [])
            rows = []
            for c in candles:
                if c.get("complete", False):
                    # parse ISO timestamp, drop nanoseconds
                    ts = pd.to_datetime(c["time"].split(".")[0])
                    rows.append({
                        "datetime": ts,
                        "open":  float(c["mid"]["o"]),
                        "high":  float(c["mid"]["h"]),
                        "low":   float(c["mid"]["l"]),
                        "close": float(c["mid"]["c"]),
                        "volume": int(c.get("volume", 0))
                    })

            if not rows:
                logger.warning(f"No completed candles returned for {instrument}")
                return pd.DataFrame()

            df = pd.DataFrame(rows).set_index("datetime")
            logger.info(f"Fetched {len(df)} complete {granularity} candles for {instrument}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OANDA data for {instrument}: {e}")
            return pd.DataFrame()
