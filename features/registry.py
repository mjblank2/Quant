from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

@dataclass
class FeatureDefinition:
    name: str
    description: str
    feature_type: str  # 'price' | 'fundamental' | 'technical' | 'alternative'
    computation: Callable[[pd.DataFrame], pd.Series]
    dependencies: List[str] = field(default_factory=list)
    lookback_days: int = 1
    update_frequency: str = "daily"
    data_sources: List[str] = field(default_factory=list)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class FeatureRegistry:
    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._register_core()

    def register(self, fd: FeatureDefinition) -> None:
        if fd.name in self._features:
            log.warning("Overwriting feature: %s", fd.name)
        self._features[fd.name] = fd

    def get(self, name: str) -> Optional[FeatureDefinition]:
        return self._features.get(name)

    def list_names(self) -> List[str]:
        return list(self._features.keys())

    def get_dependencies(self, name: str) -> List[str]:
        f = self.get(name)
        if not f:
            return []
        deps = list(f.dependencies)
        for d in f.dependencies:
            deps.extend(self.get_dependencies(d))
        # dedupe while preserving order
        out: List[str] = []
        for x in deps:
            if x not in out:
                out.append(x)
        return out

    # -------------------- core features --------------------
    def _register_core(self) -> None:
        self.register(FeatureDefinition(
            name="ret_1d",
            description="1-day return",
            feature_type="price",
            computation=lambda df: df["price_feat"].pct_change(1),
            data_sources=["daily_bars"],
            tags=["returns","momentum"],
        ))
        self.register(FeatureDefinition(
            name="ret_5d",
            description="5-day return",
            feature_type="price",
            computation=lambda df: df["price_feat"].pct_change(5),
            data_sources=["daily_bars"],
            lookback_days=5,
            tags=["returns","momentum"],
        ))
        self.register(FeatureDefinition(
            name="ret_21d",
            description="21-day return",
            feature_type="price",
            computation=lambda df: df["price_feat"].pct_change(21),
            data_sources=["daily_bars"],
            lookback_days=21,
            tags=["returns","momentum"],
        ))
        self.register(FeatureDefinition(
            name="vol_21",
            description="21-day rolling volatility (not annualized)",
            feature_type="technical",
            computation=lambda df: df["ret_1d"].rolling(21).std(),
            dependencies=["ret_1d"],
            lookback_days=21,
            tags=["volatility","risk"],
        ))
        self.register(FeatureDefinition(
            name="mom_21",
            description="21-day momentum",
            feature_type="technical",
            computation=lambda df: (df["price_feat"] / df["price_feat"].shift(21)) - 1.0,
            data_sources=["daily_bars"],
            lookback_days=21,
            tags=["momentum"],
        ))
        self.register(FeatureDefinition(
            name="mom_63",
            description="63-day momentum",
            feature_type="technical",
            computation=lambda df: (df["price_feat"] / df["price_feat"].shift(63)) - 1.0,
            data_sources=["daily_bars"],
            lookback_days=63,
            tags=["momentum"],
        ))
        self.register(FeatureDefinition(
            name="rsi_14",
            description="Relative Strength Index (14)",
            feature_type="technical",
            computation=lambda df: _rsi(df["price_feat"], 14),
            data_sources=["daily_bars"],
            lookback_days=14,
            tags=["momentum","overbought/oversold"],
        ))
        self.register(FeatureDefinition(
            name="overnight_gap",
            description="Open-to-prior-close gap",
            feature_type="microstructure",
            computation=lambda df: (df["open"] / df["price_feat"].shift(1)) - 1.0,
            data_sources=["daily_bars"],
            tags=["microstructure"],
        ))
        self.register(FeatureDefinition(
            name="adv_usd_21",
            description="21-day average dollar volume",
            feature_type="liquidity",
            computation=lambda df: (df["price_feat"] * df["volume"]).rolling(21).mean(),
            data_sources=["daily_bars"],
            lookback_days=21,
            tags=["liquidity"],
        ))
        self.register(FeatureDefinition(
            name="illiq_21",
            description="Amihud illiquidity 21d",
            feature_type="liquidity",
            computation=lambda df: (df["ret_1d"].abs() / (df["price_feat"]*df["volume"]).replace(0, np.nan)).rolling(21).mean(),
            dependencies=["ret_1d"],
            data_sources=["daily_bars"],
            lookback_days=21,
            tags=["liquidity"],
        ))
        self.register(FeatureDefinition(
            name="size_ln",
            description="Log market capitalization (PIT)",
            feature_type="fundamental",
            computation=lambda df: np.log((df["price_feat"] * (df["shares_out"] if "shares_out" in df.columns else 1.0)).astype(float).clip(lower=1.0)),
            data_sources=["daily_bars","shares_outstanding"],
            tags=["size","fundamental"],
        ))
        self.register(FeatureDefinition(
            name="turnover_21",
            description="Turnover 21d = ADV / Market Cap",
            feature_type="liquidity",
            computation=lambda df: ((df["price_feat"] * df["volume"]).rolling(21).mean()) / (df["price_feat"] * (df["shares_out"] if "shares_out" in df.columns else 1.0)).replace(0,np.nan),
            data_sources=["daily_bars","shares_outstanding"],
            lookback_days=21,
            tags=["liquidity"],
        ))

def _rsi(prices: pd.Series, window: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))

# Global registry instance
registry = FeatureRegistry()
