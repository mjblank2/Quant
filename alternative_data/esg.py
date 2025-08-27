"""
ESG Data Processing for Alternative Alpha
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import logging

log = logging.getLogger(__name__)

class ESGDataProcessor:
    """Process ESG (Environmental, Social, Governance) data for alpha generation"""
    
    def __init__(self):
        self.esg_weights = {
            'environmental': 0.4,
            'social': 0.3,
            'governance': 0.3
        }
        
    def calculate_esg_scores(self, symbols: List[str]) -> pd.DataFrame:
        """
        Calculate comprehensive ESG scores for given symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            DataFrame with ESG scores and components
        """
        esg_data = []
        
        for symbol in symbols:
            # In practice, this would come from ESG data providers
            # like MSCI, Sustainalytics, Refinitiv, Bloomberg ESG
            esg_metrics = self._get_mock_esg_data(symbol)
            
            # Calculate component scores
            env_score = self._calculate_environmental_score(esg_metrics)
            social_score = self._calculate_social_score(esg_metrics)
            governance_score = self._calculate_governance_score(esg_metrics)
            
            # Overall ESG score
            overall_score = (
                self.esg_weights['environmental'] * env_score +
                self.esg_weights['social'] * social_score +
                self.esg_weights['governance'] * governance_score
            )
            
            esg_data.append({
                'symbol': symbol,
                'esg_score': overall_score,
                'environmental_score': env_score,
                'social_score': social_score,
                'governance_score': governance_score,
                **esg_metrics  # Include raw metrics
            })
        
        return pd.DataFrame(esg_data)
    
    def _get_mock_esg_data(self, symbol: str) -> Dict[str, float]:
        """Get mock ESG data (in practice, from ESG data providers)"""
        # Generate deterministic but varied mock data based on symbol
        np.random.seed(hash(symbol) % 2**32)
        
        # Environmental metrics
        carbon_emissions = np.random.uniform(50, 500)  # tons CO2/year
        water_usage = np.random.uniform(100, 1000)     # liters per unit
        waste_recycling = np.random.uniform(0.2, 0.9)  # recycling rate
        renewable_energy = np.random.uniform(0.1, 0.8) # renewable energy %
        
        # Social metrics
        employee_satisfaction = np.random.uniform(3.0, 5.0)  # 1-5 scale
        diversity_ratio = np.random.uniform(0.3, 0.7)        # diverse workforce %
        safety_incidents = np.random.uniform(0, 10)          # incidents per year
        community_investment = np.random.uniform(0.5, 3.0)   # % of revenue
        
        # Governance metrics
        board_independence = np.random.uniform(0.4, 0.9)     # independent directors %
        exec_compensation_ratio = np.random.uniform(50, 300) # CEO to median pay ratio
        transparency_score = np.random.uniform(60, 95)       # transparency index
        audit_quality = np.random.uniform(3.0, 5.0)         # audit quality score
        
        return {
            # Environmental
            'carbon_emissions': carbon_emissions,
            'water_usage': water_usage, 
            'waste_recycling': waste_recycling,
            'renewable_energy': renewable_energy,
            
            # Social
            'employee_satisfaction': employee_satisfaction,
            'diversity_ratio': diversity_ratio,
            'safety_incidents': safety_incidents,
            'community_investment': community_investment,
            
            # Governance
            'board_independence': board_independence,
            'exec_compensation_ratio': exec_compensation_ratio,
            'transparency_score': transparency_score,
            'audit_quality': audit_quality
        }
    
    def _calculate_environmental_score(self, metrics: Dict[str, float]) -> float:
        """Calculate environmental component score (0-100)"""
        # Lower emissions = better score
        emissions_score = max(0, 100 - metrics['carbon_emissions'] / 5)
        
        # Lower water usage = better score  
        water_score = max(0, 100 - metrics['water_usage'] / 10)
        
        # Higher recycling = better score
        recycling_score = metrics['waste_recycling'] * 100
        
        # Higher renewable energy = better score
        renewable_score = metrics['renewable_energy'] * 100
        
        # Weighted average
        env_score = (
            0.3 * emissions_score +
            0.2 * water_score +
            0.25 * recycling_score +
            0.25 * renewable_score
        )
        
        return np.clip(env_score, 0, 100)
    
    def _calculate_social_score(self, metrics: Dict[str, float]) -> float:
        """Calculate social component score (0-100)"""
        # Employee satisfaction (scale to 0-100)
        satisfaction_score = (metrics['employee_satisfaction'] - 1) / 4 * 100
        
        # Diversity ratio (already in %)
        diversity_score = metrics['diversity_ratio'] * 100
        
        # Safety (lower incidents = better score)
        safety_score = max(0, 100 - metrics['safety_incidents'] * 10)
        
        # Community investment
        community_score = min(100, metrics['community_investment'] / 3 * 100)
        
        # Weighted average
        social_score = (
            0.3 * satisfaction_score +
            0.25 * diversity_score +
            0.25 * safety_score +
            0.2 * community_score
        )
        
        return np.clip(social_score, 0, 100)
    
    def _calculate_governance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate governance component score (0-100)"""
        # Board independence
        independence_score = metrics['board_independence'] * 100
        
        # Executive compensation (lower ratio = better score)
        comp_score = max(0, 100 - (metrics['exec_compensation_ratio'] - 50) / 250 * 100)
        
        # Transparency
        transparency_score = metrics['transparency_score']
        
        # Audit quality (scale to 0-100)
        audit_score = (metrics['audit_quality'] - 1) / 4 * 100
        
        # Weighted average
        governance_score = (
            0.3 * independence_score +
            0.25 * comp_score +
            0.25 * transparency_score +
            0.2 * audit_score
        )
        
        return np.clip(governance_score, 0, 100)
    
    def calculate_esg_momentum(self, historical_esg: pd.DataFrame) -> pd.DataFrame:
        """Calculate ESG momentum and trends"""
        if historical_esg.empty:
            return pd.DataFrame()
        
        momentum_data = []
        
        for symbol, group in historical_esg.groupby('symbol'):
            if len(group) < 4:  # Need at least 4 quarters
                continue
            
            group = group.sort_values('date')
            
            # Calculate trends
            esg_trend = self._calculate_trend(group['esg_score'].values)
            env_trend = self._calculate_trend(group['environmental_score'].values)
            social_trend = self._calculate_trend(group['social_score'].values)
            governance_trend = self._calculate_trend(group['governance_score'].values)
            
            # ESG volatility
            esg_volatility = group['esg_score'].std()
            
            # Recent vs historical performance
            recent_score = group['esg_score'].tail(2).mean()
            historical_score = group['esg_score'].head(-2).mean()
            esg_improvement = recent_score - historical_score
            
            momentum_data.append({
                'symbol': symbol,
                'esg_trend': esg_trend,
                'environmental_trend': env_trend,
                'social_trend': social_trend,
                'governance_trend': governance_trend,
                'esg_volatility': esg_volatility,
                'esg_improvement': esg_improvement,
                'latest_esg_score': group['esg_score'].iloc[-1]
            })
        
        return pd.DataFrame(momentum_data)
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate linear trend of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def generate_esg_alpha_signals(self, esg_scores: pd.DataFrame, 
                                 price_data: pd.DataFrame,
                                 market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate alpha signals based on ESG analysis
        
        Args:
            esg_scores: ESG scores DataFrame
            price_data: Historical price data
            market_data: Additional market data (optional)
            
        Returns:
            DataFrame with ESG-based alpha signals
        """
        if esg_scores.empty:
            return pd.DataFrame()
        
        alpha_signals = []
        
        for _, esg_row in esg_scores.iterrows():
            symbol = esg_row['symbol']
            
            # ESG quality signal
            esg_quality_signal = self._calculate_esg_quality_signal(esg_row)
            
            # ESG momentum signal (if trend data available)
            esg_momentum_signal = esg_row.get('esg_trend', 0.0) / 100  # Normalize
            
            # ESG controversy signal (sudden drops in score)
            controversy_signal = self._detect_esg_controversies(esg_row)
            
            # Combine signals
            combined_alpha = (
                0.5 * esg_quality_signal +
                0.3 * esg_momentum_signal +
                0.2 * controversy_signal
            )
            
            # Adjust for market conditions if available
            if market_data is not None and not market_data.empty:
                market_adjustment = self._adjust_for_market_conditions(symbol, market_data)
                combined_alpha *= market_adjustment
            
            alpha_signals.append({
                'symbol': symbol,
                'esg_alpha': combined_alpha,
                'esg_quality_signal': esg_quality_signal,
                'esg_momentum_signal': esg_momentum_signal,
                'esg_controversy_signal': controversy_signal,
                'esg_score': esg_row['esg_score'],
                'confidence': min(0.7, esg_row.get('latest_esg_score', 50) / 100)
            })
        
        return pd.DataFrame(alpha_signals)
    
    def _calculate_esg_quality_signal(self, esg_row: pd.Series) -> float:
        """Calculate alpha signal based on ESG quality"""
        esg_score = esg_row['esg_score']
        
        # High ESG scores get positive signal (ESG premium)
        if esg_score >= 80:
            return 0.2   # Strong positive signal
        elif esg_score >= 60:
            return 0.1   # Moderate positive signal
        elif esg_score >= 40:
            return 0.0   # Neutral
        elif esg_score >= 20:
            return -0.1  # Moderate negative signal
        else:
            return -0.2  # Strong negative signal
    
    def _detect_esg_controversies(self, esg_row: pd.Series) -> float:
        """Detect ESG controversies from sudden score changes"""
        # This would normally analyze time series data
        # For now, use improvement metric if available
        improvement = esg_row.get('esg_improvement', 0.0)
        
        if improvement < -10:  # Significant deterioration
            return -0.3
        elif improvement < -5:
            return -0.1
        elif improvement > 10:  # Significant improvement
            return 0.2
        elif improvement > 5:
            return 0.1
        else:
            return 0.0
    
    def _adjust_for_market_conditions(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Adjust ESG alpha for market conditions"""
        # ESG factors tend to outperform in certain market conditions
        # This is a simplified adjustment
        
        # In risk-off environments, ESG quality is more valued
        market_volatility = market_data.get('market_volatility', 0.2)
        
        if market_volatility > 0.3:  # High volatility = risk-off
            return 1.2  # Boost ESG signal
        elif market_volatility < 0.15:  # Low volatility = risk-on
            return 0.8  # Reduce ESG signal
        else:
            return 1.0  # Neutral adjustment