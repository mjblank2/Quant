
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

@dataclass
class FeatureDefinition:
    name: str
    description: str
    feature_type: str  # 'price', 'fundamental', 'technical', 'alternative'
    computation: Callable[[pd.DataFrame], pd.Series]
    dependencies: List[str] = field(default_factory=list)
    lookback_days: int = 1
    update_frequency: str = 'daily'
    data_sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)

class FeatureRegistry:
    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._register_core_features()

    def register_feature(self, feature_def: FeatureDefinition) -> None:
        if feature_def.name in self._features:
            log.warning(f"Overwriting existing feature definition: {feature_def.name}")
        self._features[feature_def.name] = feature_def
        log.info(f"Registered feature: {feature_def.name} (v{feature_def.version})")

    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        return self._features.get(name)

    def list_features(self, feature_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[FeatureDefinition]:
        features = list(self._features.values())
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]
        if tags:
            features = [f for f in features if any(tag in f.tags for tag in tags)]
        return features

    def get_dependencies(self, feature_name: str) -> List[str]:
        feature = self.get_feature(feature_name)
        if not feature:
            return []
        deps = set(feature.dependencies)
        for dep in feature.dependencies:
            deps.update(self.get_dependencies(dep))
        return list(deps)

    def validate_dependencies(self) -> Dict[str, List[str]]:
        errors = {}
        for name, feature in self._features.items():
            missing_deps = [dep for dep in feature.dependencies if dep not in self._features]
            if missing_deps:
                errors[name] = missing_deps
        return errors

    def _register_core_features(self):
        self.register_feature(FeatureDefinition(
            name="ret_1d",
            description="1-day return",
            feature_type="price",
            computation=lambda df: df['price_feat'].pct_change(1),
            data_sources=["daily_bars"],
            tags=["returns", "momentum"]
        ))
        self.register_feature(FeatureDefinition(
            name="ret_5d",
            description="5-day return",
            feature_type="price",
            computation=lambda df: df['price_feat'].pct_change(5),
            data_sources=["daily_bars"],
            lookback_days=5,
            tags=["returns", "momentum"]
        ))
        self.register_feature(FeatureDefinition(
            name="vol_21",
            description="21-day rolling volatility (annualized)",
            feature_type="technical",
            computation=lambda df: df['ret_1d'].rolling(21).std() * (252**0.5),
            dependencies=["ret_1d"],
            lookback_days=21,
            tags=["volatility", "risk"]
        ))
        self.register_feature(FeatureDefinition(
            name="mom_21",
            description="21-day momentum",
            feature_type="technical",
            computation=lambda df: df['price_feat'].pct_change(21),
            data_sources=["daily_bars"],
            lookback_days=21,
            tags=["momentum"]
        ))
        self.register_feature(FeatureDefinition(
            name="size_ln",
            description="Log market capitalization",
            feature_type="fundamental",
            computation=lambda df: np.log((df['price_feat'] * df.get('shares_out', 1)).clip(lower=1.0)),
            data_sources=["daily_bars", "shares_outstanding"],
            tags=["size", "fundamental"]
        ))
        self.register_feature(FeatureDefinition(
            name="turnover_21",
            description="21-day average turnover ratio",
            feature_type="technical",
            computation=lambda df: (df['volume'] * df['price_feat']).rolling(21).mean() / (df['price_feat'] * df.get('shares_out', 1)),
            data_sources=["daily_bars", "shares_outstanding"],
            lookback_days=21,
            tags=["liquidity", "volume"]
        ))

registry = FeatureRegistry()
