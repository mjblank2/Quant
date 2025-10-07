"""
Price utilities for handling optional adj_close column in daily_bars table.

This module provides centralized utilities to detect the presence of the adj_close column
in the daily_bars table and return appropriate SQL expressions for price queries.

All price-related SQL queries throughout the codebase use these utilities to ensure
resilient operation with or without the adj_close column, eliminating hard dependencies
and preventing psycopg.errors.UndefinedColumn crashes.

Key functions:
- price_expr(): Returns "COALESCE(adj_close, close)" or "close" dynamically
- select_price_as(alias): Returns price expression with custom column alias
- has_adj_close(): Detects column presence (cached for performance)
"""
from functools import lru_cache
from sqlalchemy import inspect
from db import engine
import logging

log = logging.getLogger(__name__)

# Guard to ensure we only log the adj_close mode once
_logged_price_mode = False
_price_expr_cache: str | None = None
_price_expr_state: tuple[bool, str] | None = None


@lru_cache(maxsize=1)
def has_adj_close() -> bool:
    """
    Check if the daily_bars table has an adj_close column.

    Returns
    -------
    bool
        True if adj_close column exists, False otherwise.
    """
    try:
        insp = inspect(engine)
        columns = insp.get_columns('daily_bars')
        return any(c['name'] == 'adj_close' for c in columns)
    except Exception as e:
        log.warning(f"Column inspection failed, assuming no adj_close: {e}")
        return False


def price_expr() -> str:
    """
    Get the appropriate SQL expression for fetching adjusted prices.

    Returns
    -------
    str
        SQL expression: "COALESCE(adj_close, close)" if adj_close column exists,
        otherwise just "close".
    """
    global _logged_price_mode

    global _price_expr_cache, _price_expr_state

    state = has_adj_close()
    engine_signature = _engine_signature()

    if _price_expr_cache is not None and _price_expr_state == (state, engine_signature):
        return _price_expr_cache

    if state:
        if not _logged_price_mode:
            log.info("Using adj_close column with fallback to close")
            _logged_price_mode = True
        expression = "COALESCE(adj_close, close)"
    else:
        if not _logged_price_mode:
            log.info("adj_close column not found, falling back to close prices")
            _logged_price_mode = True
        expression = "close"

    _price_expr_cache = expression
    _price_expr_state = (state, engine_signature)
    return expression


def select_price_as(alias: str) -> str:
    """
    Get SQL expression for price selection with custom alias.

    Parameters
    ----------
    alias : str
        The column alias to use for the price expression.

    Returns
    -------
    str
        SQL expression like "COALESCE(adj_close, close) AS alias" or "close AS alias"
    """
    return f"{price_expr()} AS {alias}"


def log_price_mode_once() -> None:
    """Force logging of price mode if not already logged."""
    if not _logged_price_mode:
        # This will trigger the logging in price_expr()
        price_expr()


def _engine_signature() -> str:
    try:
        return str(engine.url)
    except Exception:  # pragma: no cover - engine may not have URL
        return repr(engine)


def _clear_price_expr_cache() -> None:
    global _price_expr_cache, _price_expr_state, _logged_price_mode
    _price_expr_cache = None
    _price_expr_state = None
    _logged_price_mode = False


price_expr.cache_clear = _clear_price_expr_cache  # type: ignore[attr-defined]
