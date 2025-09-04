from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import date
import numpy as np
import pandas as pd
from sqlalchemy import text
import logging

from .registry import registry, FeatureRegistry

log = logging.getLogger(__name__)

class FeatureStore:
    """Centralized feature computation with training/serving parity."""
    def __init__(self, engine, feature_registry: Optional[FeatureRegistry] = None):
        self.engine = engine
        self.registry = feature_registry or registry

    # -------------------- public API --------------------
    def compute_features(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

        if feature_names is None:
            feature_names = self.registry.list_names()

        # expand with dependencies & topo order
        feature_names = self._topo_order(feature_names)

        base = self._load_base_data(symbols, start_date, end_date)
        if base.empty:
            return pd.DataFrame()

        out: List[pd.DataFrame] = []
        for sym, sdf in base.groupby("symbol"):
            sdf = sdf.sort_values("ts").copy()
            out.append(self._compute_for_symbol(sdf, feature_names))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def get_latest_features(self, symbols: List[str], feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        latest = self._get_latest_date()
        if latest is None:
            return pd.DataFrame()
        return self.compute_features(symbols, start_date=latest, end_date=latest, feature_names=feature_names)

    # -------------------- internals --------------------
    def _topo_order(self, names: List[str]) -> List[str]:
        needed: List[str] = []
        def add(n: str):
            if n in needed: return
            f = self.registry.get(n)
            if not f:
                return
            for d in f.dependencies:
                add(d)
            needed.append(n)
        for n in names:
            add(n)
        return needed

    def _load_base_data(self, symbols: List[str], start_date: Optional[date], end_date: Optional[date]) -> pd.DataFrame:
        # Build param list
        sym_params = {f"sym_{i}": s for i, s in enumerate(symbols)}
        sym_placeholders = ", ".join([f":sym_{i}" for i in range(len(symbols))])

        date_filter = ""
        params: Dict[str, Any] = {}
        params.update(sym_params)
        if start_date:
            date_filter += " AND ts >= :start_date"
            params["start_date"] = start_date
        if end_date:
            date_filter += " AND ts <= :end_date"
            params["end_date"] = end_date

        # Try adj_close first then fallback
        sql1 = f"""
            SELECT symbol, ts, open, close, COALESCE(adj_close, close) AS adj_close, volume,
                   COALESCE(adj_close, close) AS price_feat
            FROM daily_bars
            WHERE symbol IN ({sym_placeholders}){date_filter}
            ORDER BY symbol, ts
        """
        sql2 = f"""
            SELECT symbol, ts, open, close, close AS adj_close, volume, close AS price_feat
            FROM daily_bars
            WHERE symbol IN ({sym_placeholders}){date_filter}
            ORDER BY symbol, ts
        """
        try:
            df = pd.read_sql_query(text(sql1), self.engine, params=params, parse_dates=["ts"])
        except Exception as e:
            if "adj_close" in str(e) and ("no such column" in str(e) or "does not exist" in str(e)):
                df = pd.read_sql_query(text(sql2), self.engine, params=params, parse_dates=["ts"])
            else:
                raise

        if df.empty:
            return df

        # Shares outstanding
        sql_sh = f"""
            SELECT symbol, as_of, shares
            FROM shares_outstanding
            WHERE symbol IN ({sym_placeholders})
            ORDER BY symbol, as_of
        """
        sh = pd.read_sql_query(text(sql_sh), self.engine, params=params, parse_dates=["as_of"])
        if not sh.empty:
            sh_r = sh.rename(columns={"as_of": "ts_shares"})
            df = pd.merge_asof(
                df.sort_values(["symbol","ts"]),
                sh_r.sort_values(["symbol","ts_shares"]),
                left_on="ts", right_on="ts_shares", by="symbol", direction="backward",
            )
            df["shares_out"] = df["shares"]
        else:
            df["shares_out"] = np.nan
        return df

    def _compute_for_symbol(self, sdf: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        res = sdf[["symbol","ts"]].copy()
        for name in feature_names:
            f = self.registry.get(name)
            if not f:
                continue
            try:
                vals = f.computation(sdf)
                if isinstance(vals, pd.Series):
                    res[name] = vals.reindex(sdf.index).values
                else:
                    res[name] = vals
            except Exception as e:
                log.warning("Error computing %s for %s: %s", name, sdf["symbol"].iloc[0], e)
                res[name] = np.nan
        return res

    def _get_latest_date(self):
        try:
            r = pd.read_sql_query(text("SELECT MAX(ts) AS max_ts FROM daily_bars"), self.engine)
            return r["max_ts"].iloc[0] if not r.empty else None
        except Exception:
            return None
