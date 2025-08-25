from __future__ import annotations
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def _fix_db_url(url: str) -> str:
    if url and url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url and url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def _as_float(env_name: str, default: float) -> float:
    v = os.getenv(env_name)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return float(v)
    except Exception:
        logging.warning(f"Invalid float for {env_name}='{v}'. Using default: {default}")
        return default

def _as_int(env_name: str, default: int) -> int:
    v = os.getenv(env_name)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return int(v)
    except Exception:
        logging.warning(f"Invalid int for {env_name}='{v}'. Using default: {default}")
        return default

def _as_bool(env_name: str, default: bool) -> bool:
    v = os.getenv(env_name)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    return str(v).lower() in {"1","true","yes","y"}

DATABASE_URL = _fix_db_url(os.getenv("DATABASE_URL", ""))

# Providers / keys
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

# Liquidity gates (universe)
ADV_USD_MIN   = _as_float("ADV_USD_MIN", 25_000.0)
ADV_LOOKBACK  = _as_int("ADV_LOOKBACK", 20)
MIN_PRICE     = _as_float("MIN_PRICE", 1.00)
MIN_ADV_USD   = _as_float("MIN_ADV_USD", ADV_USD_MIN)

# Modeling
BACKTEST_START = os.getenv("BACKTEST_START", "2019-01-01")
TARGET_HORIZON_DAYS = _as_int("TARGET_HORIZON_DAYS", 5)

# Ensemble
PREFERRED_MODEL = os.getenv("PREFERRED_MODEL", "blend_v1")
BLEND_WEIGHTS   = os.getenv("BLEND_WEIGHTS", "xgb:0.5,rf:0.3,ridge:0.2")

# Portfolio construction
TOP_N = _as_int("TOP_N", 50)
ALLOW_SHORTS = _as_bool("ALLOW_SHORTS", False)
LONG_TOP_N = _as_int("LONG_TOP_N", TOP_N)
SHORT_TOP_N = _as_int("SHORT_TOP_N", TOP_N)
GROSS_LEVERAGE = _as_float("GROSS_LEVERAGE", 1.0)
_default_net = 0.0 if ALLOW_SHORTS else GROSS_LEVERAGE
NET_EXPOSURE = _as_float("NET_EXPOSURE", _default_net)
MAX_POSITION_WEIGHT = _as_float("MAX_POSITION_WEIGHT", 0.03)

# Selection/cardinality (v16)
LONG_COUNT_MIN = _as_int("LONG_COUNT_MIN", 10)
LONG_COUNT_MAX = _as_int("LONG_COUNT_MAX", 20)
SHORT_COUNT_MAX = _as_int("SHORT_COUNT_MAX", 10)

# Execution / TCA
MARKET_IMPACT_BETA = _as_float("MARKET_IMPACT_BETA", 0.1)
EXECUTION_STYLE    = os.getenv("EXECUTION_STYLE", "moc").lower()  # market | limit | moc
SPREAD_BPS         = _as_float("SPREAD_BPS", 7.0)
COMMISSION_BPS     = _as_float("COMMISSION_BPS", 0.5)

# Risk / Neutralization
SECTOR_NEUTRALIZE  = _as_bool("SECTOR_NEUTRALIZE", True)
MAX_PER_SECTOR     = _as_int("MAX_PER_SECTOR", 3)
BETA_HEDGE_SYMBOL  = os.getenv("BETA_HEDGE_SYMBOL", "IWM")
BETA_HEDGE_MAX_WEIGHT = _as_float("BETA_HEDGE_MAX_WEIGHT", 0.20)
BETA_TARGET        = _as_float("BETA_TARGET", 0.0)

# Borrow/short (foundation)
BORROW_FEE_CEILING_BPS = _as_float("BORROW_FEE_CEILING_BPS", 1000.0)
BORROW_NOLOCATE_SCALE  = _as_float("BORROW_NOLOCATE_SCALE", 1.0)
BORROW_FEE_CARRY       = _as_bool("BORROW_FEE_CARRY", True)

# Optimizer / crowding / turnover (v16)
TURNOVER_TARGET_ANNUAL = _as_float("TURNOVER_TARGET_ANNUAL", 2.5)
TURNOVER_PENALTY_BPS   = _as_float("TURNOVER_PENALTY_BPS", 25.0)
MAX_NAME_CORR          = _as_float("MAX_NAME_CORR", 0.85)

# Tax lots
TAX_LOT_METHOD     = os.getenv("TAX_LOT_METHOD", "hifo").lower()
TAX_ST_PENALTY_BPS = _as_float("TAX_ST_PENALTY_BPS", 150.0)
TAX_LT_DAYS        = _as_int("TAX_LT_DAYS", 365)
TAX_WASH_DAYS      = _as_int("TAX_WASH_DAYS", 30)

# State / capital
STARTING_CAPITAL = _as_float("STARTING_CAPITAL", 100_000.0)

# v17 fast-follow toggles
USE_UNIVERSE_HISTORY = _as_bool("USE_UNIVERSE_HISTORY", True)
REGIME_GATING        = _as_bool("REGIME_GATING", True)
USE_QP_OPTIMIZER     = _as_bool("USE_QP_OPTIMIZER", False)
QP_CORR_PENALTY      = _as_float("QP_CORR_PENALTY", 0.05)

# Russell expansion controls when snapshotting the universe
UNIVERSE_FILTER_RUSSELL = _as_bool("UNIVERSE_FILTER_RUSSELL", False)
RUSSELL_INDEX           = os.getenv("RUSSELL_INDEX", "R2000")

# Options overlay
IV_FALLBACK = _as_float("IV_FALLBACK", 0.35)
