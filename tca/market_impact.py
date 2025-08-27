"""
Market Impact Models - Advanced market impact estimation beyond simple BPS costs
Implements the square-root law and other sophisticated models used by institutional traders
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)

@dataclass
class MarketImpactParams:
    """Parameters for market impact models"""
    permanent_impact_coeff: float = 0.1  # Permanent impact coefficient
    temporary_impact_coeff: float = 0.05  # Temporary impact coefficient
    volatility_scaling: float = 1.0       # Volatility scaling factor
    adv_scaling: float = 1.0             # ADV scaling factor
    timing_risk_coeff: float = 0.02      # Timing risk coefficient

class MarketImpactModel(ABC):
    """Abstract base class for market impact models"""
    
    @abstractmethod
    def estimate_impact(self, order_size: float, adv: float, volatility: float, 
                       price: float, **kwargs) -> Dict[str, float]:
        """
        Estimate market impact for a given order
        
        Args:
            order_size: Order size in shares (signed: positive for buy, negative for sell)
            adv: Average daily volume in shares
            volatility: Daily volatility (e.g., 20-day rolling)
            price: Current price
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with impact estimates: {'permanent_bps', 'temporary_bps', 'total_bps'}
        """
        pass

class SquareRootLaw(MarketImpactModel):
    """
    Square-Root Law Market Impact Model
    Based on the empirical finding that market impact scales as the square root of order size
    
    Impact = σ * (Q/V)^α * f(timing, market_conditions)
    where:
    - σ is volatility
    - Q is order size
    - V is average daily volume
    - α ≈ 0.5 (square root scaling)
    """
    
    def __init__(self, params: Optional[MarketImpactParams] = None):
        self.params = params or MarketImpactParams()
        
    def estimate_impact(self, order_size: float, adv: float, volatility: float, 
                       price: float, participation_rate: Optional[float] = None, 
                       time_horizon_hours: float = 1.0, **kwargs) -> Dict[str, float]:
        """
        Estimate market impact using square-root law
        
        Args:
            order_size: Order size in shares (signed)
            adv: Average daily volume in shares  
            volatility: Daily volatility (annualized)
            price: Current price
            participation_rate: Target participation rate (0.0 to 1.0)
            time_horizon_hours: Time horizon for execution in hours
            
        Returns:
            Impact estimates in basis points
        """
        if adv <= 0 or volatility <= 0 or price <= 0:
            return {'permanent_bps': 0.0, 'temporary_bps': 0.0, 'total_bps': 0.0}
        
        # Convert to absolute order size for calculations
        abs_order_size = abs(order_size)
        
        # Participation rate (default to reasonable estimate if not provided)
        if participation_rate is None:
            # Estimate participation rate based on order size and horizon
            daily_execution_rate = min(0.20, abs_order_size / adv)  # Cap at 20% of ADV
            participation_rate = daily_execution_rate * (24.0 / time_horizon_hours)
            participation_rate = min(participation_rate, 0.50)  # Cap at 50%
        
        # Market impact components
        
        # 1. Permanent Impact (price discovery component)
        # Scales with participation rate and volatility
        permanent_impact = (
            self.params.permanent_impact_coeff 
            * volatility 
            * np.sqrt(participation_rate)
            * self.params.volatility_scaling
        )
        
        # 2. Temporary Impact (liquidity consumption)
        # Square-root scaling with order size relative to ADV
        order_rate = abs_order_size / adv
        temporary_impact = (
            self.params.temporary_impact_coeff
            * volatility
            * np.power(order_rate, 0.5)  # Square root law
            * self.params.adv_scaling
        )
        
        # 3. Timing Risk (depends on execution horizon)
        # Longer horizons = more timing risk but less market impact
        timing_multiplier = np.sqrt(time_horizon_hours / 24.0)  # Normalize to daily
        timing_risk = (
            self.params.timing_risk_coeff
            * volatility
            * timing_multiplier
        )
        
        # Convert to basis points
        permanent_bps = permanent_impact * 10000
        temporary_bps = (temporary_impact + timing_risk) * 10000
        total_bps = permanent_bps + temporary_bps
        
        return {
            'permanent_bps': permanent_bps,
            'temporary_bps': temporary_bps, 
            'total_bps': total_bps,
            'participation_rate': participation_rate,
            'order_rate': order_rate
        }
    
    def estimate_slippage_schedule(self, order_size: float, adv: float, volatility: float,
                                 price: float, num_slices: int = 10, 
                                 time_horizon_hours: float = 6.5) -> pd.DataFrame:
        """
        Estimate slippage for a TWAP/VWAP execution schedule
        
        Args:
            order_size: Total order size in shares
            adv: Average daily volume
            volatility: Daily volatility
            price: Current price
            num_slices: Number of execution slices
            time_horizon_hours: Total execution time
            
        Returns:
            DataFrame with slippage estimates per slice
        """
        slice_size = order_size / num_slices
        slice_duration = time_horizon_hours / num_slices
        
        results = []
        cumulative_impact = 0.0
        
        for i in range(num_slices):
            # Each slice has same size but impact accumulates
            slice_impact = self.estimate_impact(
                slice_size, adv, volatility, price,
                time_horizon_hours=slice_duration
            )
            
            # Permanent impact accumulates
            cumulative_impact += slice_impact['permanent_bps']
            
            results.append({
                'slice': i + 1,
                'slice_size': slice_size,
                'slice_duration_hours': slice_duration,
                'permanent_bps': slice_impact['permanent_bps'],
                'temporary_bps': slice_impact['temporary_bps'],
                'cumulative_permanent_bps': cumulative_impact,
                'total_slice_bps': slice_impact['total_bps'] + cumulative_impact
            })
        
        return pd.DataFrame(results)

class AlmgrenChrissPredictiveModel(MarketImpactModel):
    """
    Almgren-Chriss predictive market impact model
    More sophisticated than square-root law, accounts for volatility regime and liquidity
    """
    
    def __init__(self, alpha: float = 0.6, beta: float = 0.6, gamma: float = 0.8):
        self.alpha = alpha  # Temporary impact exponent  
        self.beta = beta    # Permanent impact exponent
        self.gamma = gamma  # Volatility scaling exponent
        
    def estimate_impact(self, order_size: float, adv: float, volatility: float,
                       price: float, spread_bps: float = 10.0, **kwargs) -> Dict[str, float]:
        """
        Estimate impact using Almgren-Chriss model
        
        Args:
            order_size: Order size in shares
            adv: Average daily volume
            volatility: Daily volatility
            price: Current price
            spread_bps: Bid-ask spread in basis points
            
        Returns:
            Impact estimates
        """
        if adv <= 0 or volatility <= 0 or price <= 0:
            return {'permanent_bps': 0.0, 'temporary_bps': 0.0, 'total_bps': 0.0}
            
        abs_order_size = abs(order_size)
        order_fraction = abs_order_size / adv
        
        # Spread-based temporary impact
        temporary_impact_bps = (spread_bps / 2.0) * np.power(order_fraction, self.alpha)
        
        # Volatility-based permanent impact
        vol_bps = volatility * 10000  # Convert to bps
        permanent_impact_bps = 0.1 * vol_bps * np.power(order_fraction, self.beta) * np.power(vol_bps / 100.0, self.gamma - 1)
        
        total_bps = permanent_impact_bps + temporary_impact_bps
        
        return {
            'permanent_bps': permanent_impact_bps,
            'temporary_bps': temporary_impact_bps,
            'total_bps': total_bps,
            'order_fraction': order_fraction
        }

class AdaptiveMarketImpactModel(MarketImpactModel):
    """
    Adaptive market impact model that learns from historical execution data
    """
    
    def __init__(self):
        self.historical_data: List[Dict] = []
        self.model_params = MarketImpactParams()
        self.is_calibrated = False
        
    def add_execution_data(self, order_size: float, adv: float, volatility: float,
                          price: float, realized_impact_bps: float, **metadata):
        """Add historical execution for model calibration"""
        self.historical_data.append({
            'order_size': abs(order_size),
            'adv': adv,
            'volatility': volatility,
            'price': price,
            'realized_impact_bps': realized_impact_bps,
            'order_rate': abs(order_size) / adv,
            **metadata
        })
        
    def calibrate(self) -> Dict[str, float]:
        """Calibrate model parameters from historical data"""
        if len(self.historical_data) < 10:
            log.warning("Insufficient data for calibration, using default parameters")
            return {}
            
        df = pd.DataFrame(self.historical_data)
        
        # Simple linear regression to calibrate coefficients
        # In practice, would use more sophisticated optimization
        
        from sklearn.linear_model import LinearRegression
        
        # Features: volatility, sqrt(order_rate), order_rate
        X = np.column_stack([
            df['volatility'].values,
            np.sqrt(df['order_rate'].values),
            df['order_rate'].values
        ])
        
        y = df['realized_impact_bps'].values
        
        try:
            reg = LinearRegression().fit(X, y)
            
            # Update parameters based on regression
            self.model_params.permanent_impact_coeff = max(0.01, reg.coef_[0] * 10000)
            self.model_params.temporary_impact_coeff = max(0.01, reg.coef_[1] * 10000)
            
            self.is_calibrated = True
            
            return {
                'r_squared': reg.score(X, y),
                'permanent_coeff': self.model_params.permanent_impact_coeff,
                'temporary_coeff': self.model_params.temporary_impact_coeff,
                'samples': len(df)
            }
            
        except Exception as e:
            log.error(f"Calibration failed: {e}")
            return {}
    
    def estimate_impact(self, order_size: float, adv: float, volatility: float,
                       price: float, **kwargs) -> Dict[str, float]:
        """Estimate impact using calibrated or default parameters"""
        if not self.is_calibrated and self.historical_data:
            self.calibrate()
            
        # Use square-root law with calibrated parameters
        sqrt_model = SquareRootLaw(self.model_params)
        return sqrt_model.estimate_impact(order_size, adv, volatility, price, **kwargs)