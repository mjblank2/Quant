"""
Comprehensive Transaction Cost Model
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .market_impact import MarketImpactModel, SquareRootLaw
from .execution import ExecutionParams, ExecutionStyle

log = logging.getLogger(__name__)

@dataclass
class CostComponents:
    """Breakdown of transaction cost components"""
    commission_bps: float
    spread_bps: float
    market_impact_bps: float
    timing_risk_bps: float
    borrowing_cost_bps: float = 0.0
    opportunity_cost_bps: float = 0.0
    total_bps: float = 0.0
    
    def __post_init__(self):
        self.total_bps = (
            self.commission_bps + self.spread_bps + self.market_impact_bps + 
            self.timing_risk_bps + self.borrowing_cost_bps + self.opportunity_cost_bps
        )

class TransactionCostModel:
    """
    Advanced transaction cost model with multiple cost components
    Integrates market impact, liquidity costs, and execution style optimization
    """
    
    def __init__(self,
                 commission_bps: float = 0.5,
                 min_commission: float = 1.0,
                 market_impact_model: Optional[MarketImpactModel] = None):
        self.commission_bps = commission_bps
        self.min_commission = min_commission
        self.market_impact_model = market_impact_model or SquareRootLaw()
        
        # Dynamic cost adjustments based on market conditions
        self.volatility_adjustment = True
        self.liquidity_adjustment = True
        
    def estimate_costs(self,
                      symbol: str,
                      order_size: float,
                      price: float,
                      market_data: Dict[str, float],
                      execution_params: Optional[ExecutionParams] = None) -> CostComponents:
        """
        Estimate comprehensive transaction costs
        
        Args:
            symbol: Security symbol
            order_size: Order size in shares (signed: + for buy, - for sell)
            price: Current/arrival price
            market_data: Dict with 'adv', 'volatility', 'spread_bps', etc.
            execution_params: Execution parameters
            
        Returns:
            Detailed cost breakdown
        """
        if execution_params is None:
            execution_params = ExecutionParams(ExecutionStyle.TWAP)
        
        abs_order_size = abs(order_size)
        notional = abs_order_size * price
        
        # Extract market data
        adv = market_data.get('adv', 1000000)
        volatility = market_data.get('volatility', 0.20)
        spread_bps = market_data.get('spread_bps', 10.0)
        
        # 1. Commission costs
        commission_cost = self._calculate_commission_cost(notional)
        
        # 2. Spread costs
        spread_cost = self._calculate_spread_cost(spread_bps, execution_params.style)
        
        # 3. Market impact costs
        impact_cost = self._calculate_market_impact_cost(
            abs_order_size, adv, volatility, price, execution_params
        )
        
        # 4. Timing risk
        timing_risk = self._calculate_timing_risk(
            volatility, execution_params.time_horizon_hours, execution_params.style
        )
        
        # 5. Borrowing costs (for short sales)
        borrowing_cost = self._calculate_borrowing_cost(
            order_size, symbol, market_data.get('borrow_rate_bps', 0.0)
        )
        
        # 6. Opportunity cost
        opportunity_cost = self._calculate_opportunity_cost(
            execution_params.time_horizon_hours, volatility, execution_params.urgency
        )
        
        return CostComponents(
            commission_bps=commission_cost,
            spread_bps=spread_cost,
            market_impact_bps=impact_cost,
            timing_risk_bps=timing_risk,
            borrowing_cost_bps=borrowing_cost,
            opportunity_cost_bps=opportunity_cost
        )
    
    def _calculate_commission_cost(self, notional: float) -> float:
        """Calculate commission cost in basis points"""
        commission_usd = max(self.min_commission, notional * self.commission_bps / 10000)
        return commission_usd / notional * 10000 if notional > 0 else 0.0
    
    def _calculate_spread_cost(self, spread_bps: float, execution_style: ExecutionStyle) -> float:
        """Calculate spread cost based on execution style"""
        if execution_style == ExecutionStyle.MARKET:
            # Pay half spread for market orders
            return spread_bps * 0.5
        elif execution_style in [ExecutionStyle.LIMIT]:
            # May avoid spread with passive orders
            return spread_bps * 0.1  # Capture some spread
        else:
            # TWAP/VWAP typically pay partial spread
            return spread_bps * 0.3
    
    def _calculate_market_impact_cost(self,
                                    order_size: float,
                                    adv: float,
                                    volatility: float,
                                    price: float,
                                    execution_params: ExecutionParams) -> float:
        """Calculate market impact cost"""
        impact_result = self.market_impact_model.estimate_impact(
            order_size=order_size,
            adv=adv,
            volatility=volatility,
            price=price,
            participation_rate=execution_params.participation_rate,
            time_horizon_hours=execution_params.time_horizon_hours
        )
        
        return impact_result.get('total_bps', 0.0)
    
    def _calculate_timing_risk(self,
                             volatility: float,
                             time_horizon_hours: float,
                             execution_style: ExecutionStyle) -> float:
        """Calculate timing risk cost"""
        if execution_style == ExecutionStyle.MARKET:
            return 0.0  # No timing risk for immediate execution
        
        # Timing risk scales with sqrt(time) and volatility
        daily_vol_bps = volatility * 10000
        time_factor = np.sqrt(time_horizon_hours / 24.0)  # Normalize to daily
        
        # Base timing risk coefficient
        timing_coeff = 0.1 if execution_style in [ExecutionStyle.VWAP, ExecutionStyle.TWAP] else 0.05
        
        return daily_vol_bps * time_factor * timing_coeff
    
    def _calculate_borrowing_cost(self,
                                order_size: float,
                                symbol: str,
                                borrow_rate_bps: float) -> float:
        """Calculate borrowing cost for short sales"""
        if order_size >= 0:  # Long position
            return 0.0
        
        # For short sales, include borrowing cost
        # This is typically an annual rate, need to prorate for holding period
        # Assuming average holding period of 30 days for simplicity
        holding_period_days = 30
        annual_borrow_cost = borrow_rate_bps
        prorated_cost = annual_borrow_cost * (holding_period_days / 365)
        
        return prorated_cost
    
    def _calculate_opportunity_cost(self,
                                  time_horizon_hours: float,
                                  volatility: float,
                                  urgency: float) -> float:
        """Calculate opportunity cost of delayed execution"""
        if time_horizon_hours <= 0.5:  # Less than 30 minutes
            return 0.0
        
        # Opportunity cost based on potential price movement during delay
        daily_vol_bps = volatility * 10000
        time_factor = np.sqrt(time_horizon_hours / 24.0)
        urgency_factor = urgency  # Higher urgency = higher opportunity cost
        
        return daily_vol_bps * time_factor * urgency_factor * 0.1
    
    def estimate_portfolio_costs(self,
                               trades: pd.DataFrame,
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate costs for a portfolio of trades
        
        Args:
            trades: DataFrame with columns ['symbol', 'quantity', 'price', 'execution_style']
            market_data: DataFrame with market statistics by symbol
            
        Returns:
            DataFrame with cost estimates per trade
        """
        if trades.empty:
            return pd.DataFrame()
        
        # Merge trades with market data
        trades_with_market = trades.merge(
            market_data, on='symbol', how='left'
        )
        
        cost_estimates = []
        
        for _, trade in trades_with_market.iterrows():
            symbol = trade['symbol']
            order_size = trade['quantity']
            price = trade['price']
            
            # Create execution params from trade data
            execution_style = ExecutionStyle(trade.get('execution_style', 'twap'))
            execution_params = ExecutionParams(
                style=execution_style,
                participation_rate=trade.get('participation_rate', 0.10),
                time_horizon_hours=trade.get('time_horizon_hours', 4.0),
                urgency=trade.get('urgency', 0.5)
            )
            
            # Prepare market data dict
            market_dict = {
                'adv': trade.get('adv', 1000000),
                'volatility': trade.get('volatility', 0.20),
                'spread_bps': trade.get('spread_bps', 10.0),
                'borrow_rate_bps': trade.get('borrow_rate_bps', 0.0)
            }
            
            # Estimate costs
            costs = self.estimate_costs(symbol, order_size, price, market_dict, execution_params)
            
            # Create result record
            notional = abs(order_size) * price
            cost_record = {
                'symbol': symbol,
                'quantity': order_size,
                'price': price,
                'notional': notional,
                'commission_bps': costs.commission_bps,
                'spread_bps': costs.spread_bps,
                'market_impact_bps': costs.market_impact_bps,
                'timing_risk_bps': costs.timing_risk_bps,
                'borrowing_cost_bps': costs.borrowing_cost_bps,
                'opportunity_cost_bps': costs.opportunity_cost_bps,
                'total_cost_bps': costs.total_bps,
                'total_cost_usd': notional * costs.total_bps / 10000,
                'execution_style': execution_style.value
            }
            
            cost_estimates.append(cost_record)
        
        return pd.DataFrame(cost_estimates)
    
    def benchmark_execution_styles(self,
                                 symbol: str,
                                 order_size: float,
                                 price: float,
                                 market_data: Dict[str, float]) -> pd.DataFrame:
        """
        Compare costs across different execution styles
        
        Returns:
            DataFrame comparing execution styles
        """
        styles_to_test = [
            ExecutionStyle.MARKET,
            ExecutionStyle.TWAP,
            ExecutionStyle.VWAP,
            ExecutionStyle.IMPLEMENTATION_SHORTFALL
        ]
        
        results = []
        
        for style in styles_to_test:
            # Create appropriate execution params for each style
            if style == ExecutionStyle.MARKET:
                params = ExecutionParams(style, participation_rate=1.0, time_horizon_hours=0.1)
            elif style == ExecutionStyle.TWAP:
                params = ExecutionParams(style, participation_rate=0.15, time_horizon_hours=4.0)
            elif style == ExecutionStyle.VWAP:
                params = ExecutionParams(style, participation_rate=0.20, time_horizon_hours=6.0)
            else:  # IS
                params = ExecutionParams(style, participation_rate=0.10, time_horizon_hours=2.0)
            
            costs = self.estimate_costs(symbol, order_size, price, market_data, params)
            
            results.append({
                'execution_style': style.value,
                'participation_rate': params.participation_rate,
                'time_horizon_hours': params.time_horizon_hours,
                'commission_bps': costs.commission_bps,
                'spread_bps': costs.spread_bps,
                'market_impact_bps': costs.market_impact_bps,
                'timing_risk_bps': costs.timing_risk_bps,
                'total_cost_bps': costs.total_bps
            })
        
        return pd.DataFrame(results).sort_values('total_cost_bps')
    
    def calibrate_from_fills(self, historical_fills: pd.DataFrame) -> Dict[str, float]:
        """
        Calibrate model parameters from historical execution data
        
        Args:
            historical_fills: DataFrame with execution history
            
        Returns:
            Calibrated parameters
        """
        if historical_fills.empty:
            return {}
        
        # This would implement sophisticated parameter calibration
        # For now, return basic statistics
        
        realized_costs = historical_fills.get('realized_cost_bps', pd.Series())
        if realized_costs.empty:
            return {}
        
        return {
            'avg_realized_cost_bps': realized_costs.mean(),
            'cost_volatility_bps': realized_costs.std(),
            'min_cost_bps': realized_costs.min(),
            'max_cost_bps': realized_costs.max(),
            'n_observations': len(realized_costs)
        }