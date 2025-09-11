#!/usr/bin/env python3
"""
Feature breadth validation script.

Validates symbol breadth on the most recent feature date and warns if
the count falls below configured thresholds.
"""
from __future__ import annotations
import os
import pandas as pd
from sqlalchemy import create_engine, text

# Configuration - can be overridden via environment variables
MIN_SYMBOL_THRESHOLD = int(os.getenv("FEATURE_BREADTH_MIN_SYMBOLS", "100"))
WARN_SYMBOL_THRESHOLD = int(os.getenv("FEATURE_BREADTH_WARN_SYMBOLS", "200"))

def _engine():
    """Create database engine."""
    db = os.getenv("DATABASE_URL")
    if not db:
        raise RuntimeError("DATABASE_URL environment variable is required")
    if db.startswith("postgres://"):
        db = db.replace("postgres://", "postgresql+psycopg://", 1)
    return create_engine(db)

def get_latest_feature_breadth() -> dict:
    """
    Get symbol breadth statistics for the latest feature date.

    Returns
    -------
    dict
        Dictionary with keys: ts, total_symbols, symbols_with_features, coverage_rate
    """
    engine = _engine()

    with engine.connect() as conn:
        # Get the latest feature date
        latest_date_query = text("SELECT MAX(ts) as latest_ts FROM features")
        result = conn.execute(latest_date_query).fetchone()

        if not result or not result[0]:
            return {"error": "No feature dates found"}

        latest_ts = result[0]

        # Get breadth statistics for latest date
        breadth_query = text("""
            SELECT
                COUNT(*) as total_symbols,
                COUNT(CASE WHEN ret_1d IS NOT NULL OR ret_5d IS NOT NULL
                           OR mom_21 IS NOT NULL OR vol_21 IS NOT NULL
                      THEN 1 END) as symbols_with_features
            FROM features
            WHERE ts = :latest_ts
        """)

        result = conn.execute(breadth_query, {"latest_ts": latest_ts}).fetchone()

        if result:
            total = result[0] or 0
            with_features = result[1] or 0
            coverage_rate = with_features / total if total > 0 else 0.0

            return {
                "ts": latest_ts,
                "total_symbols": total,
                "symbols_with_features": with_features,
                "coverage_rate": coverage_rate
            }
        else:
            return {"error": "No data found for latest date"}

def check_universe_breadth() -> dict:
    """
    Check universe table for total symbol count as reference.

    Returns
    -------
    dict
        Dictionary with universe symbol counts
    """
    engine = _engine()

    try:
        with engine.connect() as conn:
            universe_query = text("""
                SELECT
                    COUNT(*) as total_universe,
                    COUNT(CASE WHEN included = TRUE THEN 1 END) as included_universe
                FROM universe
            """)

            result = conn.execute(universe_query).fetchone()

            if result:
                return {
                    "total_universe": result[0] or 0,
                    "included_universe": result[1] or 0
                }
            else:
                return {"total_universe": 0, "included_universe": 0}

    except Exception as e:
        return {"error": f"Failed to check universe: {e}"}

def main():
    """Validate feature breadth and print warnings if needed."""
    try:
        print("=" * 50)
        print("FEATURE BREADTH CHECK")
        print("=" * 50)

        # Get latest feature breadth
        breadth_stats = get_latest_feature_breadth()

        if "error" in breadth_stats:
            print(f"❌ Error: {breadth_stats['error']}")
            return 1

        # Get universe reference
        universe_stats = check_universe_breadth()

        # Print current status
        print(f"\nLatest feature date: {breadth_stats['ts']}")
        print(f"Symbols with features: {breadth_stats['symbols_with_features']:,}")
        print(f"Total symbols in features table: {breadth_stats['total_symbols']:,}")
        print(f"Feature coverage rate: {breadth_stats['coverage_rate']:.1%}")

        if "error" not in universe_stats:
            print(f"\nUniverse reference:")
            print(f"  Total universe symbols: {universe_stats['total_universe']:,}")
            print(f"  Included universe symbols: {universe_stats['included_universe']:,}")

        print(f"\nThresholds:")
        print(f"  Minimum symbols: {MIN_SYMBOL_THRESHOLD:,}")
        print(f"  Warning threshold: {WARN_SYMBOL_THRESHOLD:,}")

        # Check thresholds and issue warnings
        symbols_with_features = breadth_stats['symbols_with_features']

        if symbols_with_features < MIN_SYMBOL_THRESHOLD:
            print(f"\n🚨 CRITICAL: Symbol count ({symbols_with_features:,}) is below minimum threshold ({MIN_SYMBOL_THRESHOLD:,})")
            print("   This may indicate a serious data ingestion issue.")
            return 2  # Critical error code

        elif symbols_with_features < WARN_SYMBOL_THRESHOLD:
            print(f"\n⚠️  WARNING: Symbol count ({symbols_with_features:,}) is below warning threshold ({WARN_SYMBOL_THRESHOLD:,})")
            print("   Consider investigating data ingestion completeness.")
            return 1  # Warning code

        else:
            print(f"\n✅ Symbol breadth looks healthy ({symbols_with_features:,} symbols)")

        # Additional checks
        if breadth_stats['coverage_rate'] < 0.9:
            print(f"\n⚠️  Feature coverage rate is low ({breadth_stats['coverage_rate']:.1%})")
            print("   Many symbols may be missing feature calculations.")

        # Compare to universe if available
        if "error" not in universe_stats and universe_stats['included_universe'] > 0:
            universe_coverage = symbols_with_features / universe_stats['included_universe']
            if universe_coverage < 0.8:
                print(f"\n⚠️  Low universe coverage ({universe_coverage:.1%})")
                print(f"   Only {symbols_with_features:,} of {universe_stats['included_universe']:,} universe symbols have features")

        print("\n" + "=" * 50)

        return 0

    except Exception as e:
        print(f"Error during feature breadth check: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
