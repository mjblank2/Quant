import os

def _fix_db_url(url: str) -> str:
    if url and url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url and url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = _fix_db_url(os.getenv("DATABASE_URL", ""))

# Providers / creds
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' or 'sip'
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

# Universe rules
MARKET_CAP_MAX = float(os.getenv("MARKET_CAP_MAX", 3_000_000_000))
ADV_USD_MIN = float(os.getenv("ADV_USD_MIN", 25_000))
ADV_LOOKBACK = int(os.getenv("ADV_LOOKBACK", 20))

# Modeling / backtest
BACKTEST_START = os.getenv("BACKTEST_START", "2019-01-01")
TARGET_HORIZON_DAYS = int(os.getenv("TARGET_HORIZON_DAYS", 5))

# Portfolio & trading
TOP_N = int(os.getenv("TOP_N", 50))
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "false").lower() == "true"
LONG_TOP_N = int(os.getenv("LONG_TOP_N", TOP_N))
SHORT_TOP_N = int(os.getenv("SHORT_TOP_N", TOP_N))
GROSS_LEVERAGE = float(os.getenv("GROSS_LEVERAGE", 1.0))
_default_net = 0.0 if ALLOW_SHORTS else 1.0
NET_EXPOSURE = float(os.getenv("NET_EXPOSURE", _default_net))

RISK_BUDGET = float(os.getenv("RISK_BUDGET", 100_000))
MAX_POSITION_WEIGHT = float(os.getenv("MAX_POSITION_WEIGHT", 0.03))
MIN_PRICE = float(os.getenv("MIN_PRICE", 1.00))
MIN_ADV_USD = float(os.getenv("MIN_ADV_USD", ADV_USD_MIN))

SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", 5))
