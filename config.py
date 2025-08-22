import os

def _fix_db_url(url: str) -> str:
    if url and url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url and url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def _as_float(env_name: str, default: float) -> float:
    v = os.getenv(env_name, None)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return float(v)
    except Exception:
        return default

DATABASE_URL = _fix_db_url(os.getenv("DATABASE_URL", ""))

# Vendors
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip")  # sip by default

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Universe rules
MARKET_CAP_MAX = _as_float("MARKET_CAP_MAX", 3_000_000_000.0)
ADV_USD_MIN = _as_float("ADV_USD_MIN", 25_000.0)
ADV_LOOKBACK = int(os.getenv("ADV_LOOKBACK", 20))
UNIVERSE_CONCURRENCY = int(os.getenv("UNIVERSE_CONCURRENCY", "8"))

# Modeling / backtest
BACKTEST_START = os.getenv("BACKTEST_START", "2019-01-01")
TARGET_HORIZON_DAYS = int(os.getenv("TARGET_HORIZON_DAYS", 5))

# Trading / portfolio
TOP_N = int(os.getenv("TOP_N", 50))
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "false").lower() == "true"
LONG_TOP_N = int(os.getenv("LONG_TOP_N", TOP_N))
SHORT_TOP_N = int(os.getenv("SHORT_TOP_N", TOP_N))
GROSS_LEVERAGE = _as_float("GROSS_LEVERAGE", 1.0)
_default_net = 0.0 if ALLOW_SHORTS else 1.0
NET_EXPOSURE = _as_float("NET_EXPOSURE", _default_net)

RISK_BUDGET = _as_float("RISK_BUDGET", 100_000.0)
MAX_POSITION_WEIGHT = _as_float("MAX_POSITION_WEIGHT", 0.03)
MIN_PRICE = _as_float("MIN_PRICE", 1.00)
MIN_ADV_USD = _as_float("MIN_ADV_USD", ADV_USD_MIN)

SLIPPAGE_BPS = _as_float("SLIPPAGE_BPS", 5.0)

PREFERRED_MODEL = os.getenv("PREFERRED_MODEL", "blend_v1")
BLEND_WEIGHTS = os.getenv("BLEND_WEIGHTS", "xgb:0.5,rf:0.3,ridge:0.2")

# Pipeline
PIPELINE_SYNC_BROKER = os.getenv("PIPELINE_SYNC_BROKER", "false").lower() == "true"
PIPELINE_BACKFILL_DAYS = int(os.getenv("PIPELINE_BACKFILL_DAYS", 7))

