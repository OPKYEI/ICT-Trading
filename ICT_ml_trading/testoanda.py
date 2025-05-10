from src.data_processing.oanda_data_fetcher import OandaDataFetcher
from src.utils.config import OANDA_API_TOKEN, OANDA_ACCOUNT_ID, SYMBOL

# Initialize fetcher
fetcher = OandaDataFetcher(OANDA_API_TOKEN, OANDA_ACCOUNT_ID)

# Try to fetch data
df = fetcher.fetch_ohlc(SYMBOL, granularity="H1", count=10)
print(df)