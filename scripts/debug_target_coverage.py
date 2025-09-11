#!/usr/bin/env python3
"""
Debug script to print target coverage metrics for recent dates.

Prints coverage statistics for fwd_ret and fwd_ret_resid targets
to help diagnose data quality and availability issues.
"""
from __future__ import annotations
import os
import pandas as pd
from sqlalchemy import create_engine, text

def _engine():
    """Create database engine."""
    db = os.getenv("DATABASE_URL")
    if not db:
        raise RuntimeError("DATABASE_URL environment variable is required")
    if db.startswith("postgres://"):
        db = db.replace("postgres://", "postgresql+psycopg://", 1)
    return create_engine(db)

def calculate_target_coverage(days_back: int = 30) -> pd.DataFrame:
    """
    Calculate target coverage for recent dates.

    Parameters
    ----------
    days_back : int
        Number of days back to analyze from latest feature date

    Returns
    -------
    pd.DataFrame
        Coverage metrics by date with columns: ts, total_symbols, fwd_ret_coverage, fwd_ret_resid_coverage
    """
    engine = _engine()

    with engine.connect() as conn:
        # Get recent feature dates
        try:
            # Try PostgreSQL syntax first
            query = text("""
                SELECT DISTINCT ts
                FROM features
                WHERE ts >= (SELECT MAX(ts) - INTERVAL ':days_back days' FROM features)
                ORDER BY ts DESC
            """)
            recent_dates = pd.read_sql_query(
                query, conn, params={"days_back": days_back}, parse_dates=['ts']
            )
        except Exception:
            # Fallback for SQLite which doesn't support INTERVAL
            try:
                query_sqlite = text("""
                    SELECT DISTINCT ts
                    FROM features
                    WHERE ts >= date((SELECT MAX(ts) FROM features), '-' || :days_back || ' days')
                    ORDER BY ts DESC
                """)
                recent_dates = pd.read_sql_query(
                    query_sqlite, conn, params={"days_back": days_back}, parse_dates=['ts']
                )
            except Exception:
                # Final fallback - just get all recent dates
                query_simple = text("""
                    SELECT DISTINCT ts
                    FROM features
                    ORDER BY ts DESC
                    LIMIT 30
                """)
                recent_dates = pd.read_sql_query(query_simple, conn, parse_dates=['ts'])

    if recent_dates.empty:
        print("No recent feature dates found")
        return pd.DataFrame()

    # Calculate coverage for each date
    coverage_results = []

    for _, row in recent_dates.iterrows():
        date_ts = row['ts']

        with engine.connect() as conn:
            # Get counts for this date
            coverage_query = text("""
                SELECT
                    COUNT(*) as total_symbols,
                    COUNT(fwd_ret) as fwd_ret_count,
                    COUNT(fwd_ret_resid) as fwd_ret_resid_count
                FROM features
                WHERE ts = :date_ts
            """)

            result = conn.execute(coverage_query, {"date_ts": date_ts.date()}).fetchone()

            if result and result[0] > 0:  # total_symbols > 0
                total = result[0]
                fwd_ret_count = result[1] or 0
                fwd_ret_resid_count = result[2] or 0

                coverage_results.append({
                    'ts': date_ts,
                    'total_symbols': total,
                    'fwd_ret_coverage': fwd_ret_count / total if total > 0 else 0.0,
                    'fwd_ret_resid_coverage': fwd_ret_resid_count / total if total > 0 else 0.0,
                    'fwd_ret_count': fwd_ret_count,
                    'fwd_ret_resid_count': fwd_ret_resid_count
                })

    return pd.DataFrame(coverage_results)

def main():
    """Print target coverage metrics to stdout."""
    try:
        print("=" * 60)
        print("TARGET COVERAGE DEBUG REPORT")
        print("=" * 60)

        coverage_df = calculate_target_coverage(days_back=30)

        if coverage_df.empty:
            print("No coverage data available.")
            return 0

        print(f"\nAnalyzing {len(coverage_df)} recent feature dates:\n")

        # Print detailed coverage by date
        for _, row in coverage_df.iterrows():
            print(f"Date: {row['ts'].strftime('%Y-%m-%d')}")
            print(f"  Total symbols: {row['total_symbols']:,}")
            print(f"  fwd_ret coverage: {row['fwd_ret_coverage']:.1%} ({row['fwd_ret_count']:,} symbols)")
            print(f"  fwd_ret_resid coverage: {row['fwd_ret_resid_coverage']:.1%} ({row['fwd_ret_resid_count']:,} symbols)")
            print()

        # Summary statistics
        print("SUMMARY STATISTICS:")
        print(f"  Average fwd_ret coverage: {coverage_df['fwd_ret_coverage'].mean():.1%}")
        print(f"  Average fwd_ret_resid coverage: {coverage_df['fwd_ret_resid_coverage'].mean():.1%}")
        print(f"  Min fwd_ret coverage: {coverage_df['fwd_ret_coverage'].min():.1%}")
        print(f"  Min fwd_ret_resid coverage: {coverage_df['fwd_ret_resid_coverage'].min():.1%}")
        print(f"  Average symbol count: {coverage_df['total_symbols'].mean():.0f}")
        print()

        # Coverage warnings
        low_coverage_threshold = 0.8  # 80%
        low_fwd_ret = coverage_df[coverage_df['fwd_ret_coverage'] < low_coverage_threshold]
        low_fwd_ret_resid = coverage_df[coverage_df['fwd_ret_resid_coverage'] < low_coverage_threshold]

        if not low_fwd_ret.empty:
            print(f"⚠️  WARNINGS - Low fwd_ret coverage (< {low_coverage_threshold:.0%}):")
            for _, row in low_fwd_ret.iterrows():
                print(f"    {row['ts'].strftime('%Y-%m-%d')}: {row['fwd_ret_coverage']:.1%}")
            print()

        if not low_fwd_ret_resid.empty:
            print(f"⚠️  WARNINGS - Low fwd_ret_resid coverage (< {low_coverage_threshold:.0%}):")
            for _, row in low_fwd_ret_resid.iterrows():
                print(f"    {row['ts'].strftime('%Y-%m-%d')}: {row['fwd_ret_resid_coverage']:.1%}")
            print()

        if low_fwd_ret.empty and low_fwd_ret_resid.empty:
            print("✅ All coverage levels are above threshold")

        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error analyzing target coverage: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
