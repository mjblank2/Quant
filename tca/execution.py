"""
Execution Analysis and Cost Modeling for TCA
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from .market_impact import MarketImpactModel, SquareRootLaw

log = logging.getLogger(__name__)

class ExecutionStyle(Enum):
    """Execution algorithm styles"""
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "is"
    MARKET_ON_CLOSE = "moc"

@dataclass
class ExecutionParams:
    """Parameters for execution algorithm"""
    style: ExecutionStyle
    participation_rate: float = 0.10  # Target participation rate
    time_horizon_hours: float = 6.5   # Trading hours in a day
    risk_aversion: float = 1.0        # Risk aversion parameter
    urgency: float = 0.5              # Urgency factor (0=patient, 1=urgent)

@dataclass
class FillRecord:
    """Record of an execution fill"""
    symbol: str
    timestamp: datetime
    fill_price: float
    fill_quantity: int
    side: str  # 'buy' or 'sell'
    execution_style: str
    venue: str
    arrival_price: float
    benchmark_price: float

class ExecutionAnalyzer:
    """Analyze execution performance and calculate TCA metrics"""
    
    def __init__(self, market_impact_model: Optional[MarketImpactModel] = None):
        self.market_impact_model = market_impact_model or SquareRootLaw()
        
    def analyze_execution(self, fills: List[FillRecord], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze execution performance across multiple fills
        
        Args:
            fills: List of fill records
            market_data: Market data with OHLCV and metadata
            
        Returns:
            Comprehensive execution analysis
        """
        if not fills:
            return {}
        
        # Group fills by symbol
        symbol_analyses = {}
        
        fills_df = pd.DataFrame([{
            'symbol': f.symbol,
            'timestamp': f.timestamp,
            'fill_price': f.fill_price,
            'fill_quantity': f.fill_quantity,
            'side': f.side,
            'execution_style': f.execution_style,
            'venue': f.venue,
            'arrival_price': f.arrival_price,
            'benchmark_price': f.benchmark_price
        } for f in fills])
        
        for symbol, symbol_fills in fills_df.groupby('symbol'):
            symbol_data = market_data[market_data['symbol'] == symbol] if 'symbol' in market_data.columns else market_data
            analysis = self._analyze_symbol_execution(symbol_fills, symbol_data)
            symbol_analyses[symbol] = analysis
        
        # Aggregate across symbols
        overall_analysis = self._aggregate_analyses(symbol_analyses)
        overall_analysis['by_symbol'] = symbol_analyses
        
        return overall_analysis
    
    def _analyze_symbol_execution(self, fills: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze execution for a single symbol"""
        if fills.empty or market_data.empty:
            return {}
        
        # Basic execution metrics
        total_quantity = fills['fill_quantity'].abs().sum()
        avg_fill_price = (fills['fill_price'] * fills['fill_quantity'].abs()).sum() / total_quantity
        arrival_price = fills['arrival_price'].iloc[0]  # Use first arrival price
        
        # Slippage calculations
        fill_slippage_bps = []
        arrival_slippage_bps = []
        
        for _, fill in fills.iterrows():
            # Slippage vs previous fill
            if len(fill_slippage_bps) > 0:
                prev_price = fills.iloc[len(fill_slippage_bps)-1]['fill_price']
                fill_slip = (fill['fill_price'] - prev_price) / prev_price * 10000
                if fill['side'] == 'sell':
                    fill_slip = -fill_slip
                fill_slippage_bps.append(fill_slip)
            
            # Slippage vs arrival
            arrival_slip = (fill['fill_price'] - arrival_price) / arrival_price * 10000
            if fill['side'] == 'sell':
                arrival_slip = -arrival_slip
            arrival_slippage_bps.append(arrival_slip)
        
        # Market impact analysis
        impact_analysis = self._estimate_market_impact(fills, market_data)
        
        # Timing analysis
        timing_analysis = self._analyze_execution_timing(fills, market_data)
        
        return {
            'total_quantity': total_quantity,
            'avg_fill_price': avg_fill_price,
            'arrival_price': arrival_price,
            'arrival_slippage_bps': np.mean(arrival_slippage_bps) if arrival_slippage_bps else 0.0,
            'fill_slippage_bps': np.mean(fill_slippage_bps) if fill_slippage_bps else 0.0,
            'slippage_volatility_bps': np.std(arrival_slippage_bps) if len(arrival_slippage_bps) > 1 else 0.0,
            'n_fills': len(fills),
            'execution_duration_minutes': (fills['timestamp'].max() - fills['timestamp'].min()).total_seconds() / 60,
            'market_impact': impact_analysis,
            'timing': timing_analysis
        }
    
    def _estimate_market_impact(self, fills: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate market impact of execution"""
        if fills.empty or market_data.empty:
            return {}
        
        # Get market statistics
        recent_data = market_data.tail(20)  # Last 20 days
        avg_volume = recent_data['volume'].mean() if 'volume' in recent_data.columns else 1000000
        volatility = recent_data['close'].pct_change().std() * np.sqrt(252) if 'close' in recent_data.columns else 0.20
        price = fills['arrival_price'].iloc[0]
        
        # Total order size
        total_order_size = fills['fill_quantity'].sum()  # Signed quantity
        
        # Estimate impact using model
        impact_estimate = self.market_impact_model.estimate_impact(
            order_size=abs(total_order_size),
            adv=avg_volume,
            volatility=volatility,
            price=price
        )
        
        return {
            'estimated_impact_bps': impact_estimate.get('total_bps', 0.0),
            'permanent_impact_bps': impact_estimate.get('permanent_bps', 0.0),
            'temporary_impact_bps': impact_estimate.get('temporary_bps', 0.0),
            'order_rate': abs(total_order_size) / avg_volume,
            'participation_rate': impact_estimate.get('participation_rate', 0.0)
        }
    
    def _analyze_execution_timing(self, fills: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timing of execution"""
        if fills.empty:
            return {}
        
        # Execution timing statistics
        start_time = fills['timestamp'].min()
        end_time = fills['timestamp'].max()
        duration = (end_time - start_time).total_seconds() / 3600  # Hours
        
        # Fill rate analysis
        fill_intervals = []
        if len(fills) > 1:
            fills_sorted = fills.sort_values('timestamp')
            for i in range(1, len(fills_sorted)):
                interval = (fills_sorted.iloc[i]['timestamp'] - fills_sorted.iloc[i-1]['timestamp']).total_seconds() / 60
                fill_intervals.append(interval)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration_hours': duration,
            'avg_fill_interval_minutes': np.mean(fill_intervals) if fill_intervals else 0.0,
            'fill_interval_std_minutes': np.std(fill_intervals) if len(fill_intervals) > 1 else 0.0,
            'fills_per_hour': len(fills) / max(duration, 1/60)  # Avoid division by zero
        }
    
    def _aggregate_analyses(self, symbol_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate analysis across symbols"""
        if not symbol_analyses:
            return {}
        
        # Volume-weighted averages
        total_quantity = sum(analysis.get('total_quantity', 0) for analysis in symbol_analyses.values())
        
        if total_quantity == 0:
            return {'total_quantity': 0}
        
        weighted_slippage = sum(
            analysis.get('arrival_slippage_bps', 0) * analysis.get('total_quantity', 0)
            for analysis in symbol_analyses.values()
        ) / total_quantity
        
        weighted_impact = sum(
            analysis.get('market_impact', {}).get('estimated_impact_bps', 0) * analysis.get('total_quantity', 0)
            for analysis in symbol_analyses.values()
        ) / total_quantity
        
        return {
            'total_quantity': total_quantity,
            'n_symbols': len(symbol_analyses),
            'weighted_avg_slippage_bps': weighted_slippage,
            'weighted_avg_impact_bps': weighted_impact,
            'total_fills': sum(analysis.get('n_fills', 0) for analysis in symbol_analyses.values()),
            'avg_execution_duration_minutes': np.mean([
                analysis.get('execution_duration_minutes', 0) 
                for analysis in symbol_analyses.values()
            ])
        }

class TransactionCostModel:
    """Comprehensive transaction cost model"""
    
    def __init__(self, 
                 commission_bps: float = 0.5,
                 spread_bps: float = 10.0,
                 market_impact_model: Optional[MarketImpactModel] = None):
        self.commission_bps = commission_bps
        self.spread_bps = spread_bps
        self.market_impact_model = market_impact_model or SquareRootLaw()
        
    def estimate_total_cost(self, 
                           order_size: float, 
                           price: float,
                           adv: float,
                           volatility: float,
                           execution_params: Optional[ExecutionParams] = None) -> Dict[str, float]:
        """
        Estimate total transaction cost breakdown
        
        Args:
            order_size: Order size in shares (signed)
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
            execution_params: Execution parameters
            
        Returns:
            Cost breakdown in basis points and dollars
        """
        if execution_params is None:
            execution_params = ExecutionParams(ExecutionStyle.TWAP)
        
        abs_order_size = abs(order_size)
        notional = abs_order_size * price
        
        # 1. Commission costs
        commission_cost_bps = self.commission_bps
        commission_cost_usd = notional * commission_cost_bps / 10000
        
        # 2. Spread costs (half spread for market orders, full spread for crossing)
        spread_multiplier = 0.5 if execution_params.style in [ExecutionStyle.MARKET, ExecutionStyle.MOC] else 1.0
        spread_cost_bps = self.spread_bps * spread_multiplier
        spread_cost_usd = notional * spread_cost_bps / 10000
        
        # 3. Market impact costs
        impact_result = self.market_impact_model.estimate_impact(
            order_size=abs_order_size,
            adv=adv,
            volatility=volatility,
            price=price,
            participation_rate=execution_params.participation_rate,
            time_horizon_hours=execution_params.time_horizon_hours
        )
        
        impact_cost_bps = impact_result.get('total_bps', 0.0)
        impact_cost_usd = notional * impact_cost_bps / 10000
        
        # 4. Timing risk (for non-immediate executions)
        timing_risk_bps = 0.0
        if execution_params.style not in [ExecutionStyle.MARKET]:
            # Simple timing risk model
            timing_risk_bps = volatility * 10000 * np.sqrt(execution_params.time_horizon_hours / 24) * 0.1
        
        timing_risk_usd = notional * timing_risk_bps / 10000
        
        # Total costs
        total_cost_bps = commission_cost_bps + spread_cost_bps + impact_cost_bps + timing_risk_bps
        total_cost_usd = commission_cost_usd + spread_cost_usd + impact_cost_usd + timing_risk_usd
        
        return {
            'commission_bps': commission_cost_bps,
            'commission_usd': commission_cost_usd,
            'spread_bps': spread_cost_bps,
            'spread_usd': spread_cost_usd,
            'market_impact_bps': impact_cost_bps,
            'market_impact_usd': impact_cost_usd,
            'timing_risk_bps': timing_risk_bps,
            'timing_risk_usd': timing_risk_usd,
            'total_bps': total_cost_bps,
            'total_usd': total_cost_usd,
            'notional_usd': notional
        }
    
    def optimize_execution_params(self,
                                 order_size: float,
                                 price: float, 
                                 adv: float,
                                 volatility: float,
                                 urgency: float = 0.5) -> ExecutionParams:
        """
        Optimize execution parameters to minimize expected cost
        
        Args:
            order_size: Order size in shares
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
            urgency: Urgency factor (0=patient, 1=urgent)
            
        Returns:
            Optimized execution parameters
        """
        # Simple optimization based on order characteristics
        order_rate = abs(order_size) / adv
        
        # Choose execution style based on order characteristics
        if urgency > 0.8 or order_rate < 0.01:
            # Small or urgent orders - use market
            style = ExecutionStyle.MARKET
            participation_rate = 1.0
            time_horizon = 0.1  # 6 minutes
        elif order_rate < 0.05:
            # Medium orders - use TWAP
            style = ExecutionStyle.TWAP
            participation_rate = min(0.20, 0.10 / order_rate)
            time_horizon = 2.0 + (1.0 - urgency) * 4.0  # 2-6 hours
        else:
            # Large orders - use VWAP with low participation
            style = ExecutionStyle.VWAP
            participation_rate = min(0.15, 0.05 / order_rate)
            time_horizon = 4.0 + (1.0 - urgency) * 2.5  # 4-6.5 hours
        
        return ExecutionParams(
            style=style,
            participation_rate=participation_rate,
            time_horizon_hours=time_horizon,
            urgency=urgency
        )