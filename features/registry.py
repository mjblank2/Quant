"""
Feature Registry - Manages feature definitions and metadata for consistency across training and serving
"""
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)

@dataclass
class FeatureDefinition:
    """Definition of a feature including its computation logic and metadata"""
    name: str
    description: str
    feature_type: str  # 'price', 'fundamental', 'technical', 'alternative'
    computation: Callable[[pd.DataFrame], pd.Series]
    dependencies: List[str] = field(default_factory=list)
    lookback_days: int = 1
    update_frequency: str = 'daily'  # 'intraday', 'daily', 'weekly'
    data_sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)

class FeatureRegistry:
    """Registry for managing feature definitions and ensuring consistency"""
    
    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._register_core_features()
    
    def register_feature(self, feature_def: FeatureDefinition) -> None:
        """Register a new feature definition"""
        if feature_def.name in self._features:
            log.warning(f"Overwriting existing feature definition: {feature_def.name}")
        
        self._features[feature_def.name] = feature_def
        log.info(f"Registered feature: {feature_def.name} (v{feature_def.version})")
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name"""
        return self._features.get(name)
    
    def list_features(self, feature_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[FeatureDefinition]:
        """List all features, optionally filtered by type or tags"""
        features = list(self._features.values())
        
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]
        
        if tags:
            features = [f for f in features if any(tag in f.tags for tag in tags)]
        
        return features
    
    def get_dependencies(self, feature_name: str) -> List[str]:
        """Get all dependencies for a feature (recursive)"""
        feature = self.get_feature(feature_name)
        if not feature:
            return []
        
        deps = set(feature.dependencies)
        for dep in feature.dependencies:
            deps.update(self.get_dependencies(dep))
        
        return list(deps)
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate that all feature dependencies exist"""
        errors = {}
        
        for name, feature in self._features.items():
            missing_deps = []
            for dep in feature.dependencies:
                if dep not in self._features:
                    missing_deps.append(dep)
            
            if missing_deps:
                errors[name] = missing_deps
        
        return errors
    
    def _register_core_features(self):
        """Register core financial features"""
        
        # Price/Return features
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
        
        # Volatility features
        self.register_feature(FeatureDefinition(
            name="vol_21",
            description="21-day rolling volatility",
            feature_type="technical",
            computation=lambda df: df['ret_1d'].rolling(21).std() * (252**0.5),
            dependencies=["ret_1d"],
            lookback_days=21,
            tags=["volatility", "risk"]
        ))
        
        # Momentum features
        self.register_feature(FeatureDefinition(
            name="mom_21",
            description="21-day momentum",
            feature_type="technical",
            computation=lambda df: df['price_feat'].pct_change(21),
            data_sources=["daily_bars"],
            lookback_days=21,
            tags=["momentum"]
        ))
        
        # Size feature
        self.register_feature(FeatureDefinition(
            name="size_ln",
            description="Log market capitalization",
            feature_type="fundamental",
            computation=lambda df: (df['price_feat'] * df.get('shares_out', 1)).apply(lambda x: pd.Series.log(x.replace(0, 1))),
            data_sources=["daily_bars", "shares_outstanding"],
            tags=["size", "fundamental"]
        ))
        
        # Liquidity features
        self.register_feature(FeatureDefinition(
            name="turnover_21",
            description="21-day average turnover ratio",
            feature_type="technical",
            computation=lambda df: (df['volume'] * df['price_feat']).rolling(21).mean() / (df['price_feat'] * df.get('shares_out', 1)),
            data_sources=["daily_bars", "shares_outstanding"],
            lookback_days=21,
            tags=["liquidity", "volume"]
        ))

# Global registry instance
registry = FeatureRegistry()