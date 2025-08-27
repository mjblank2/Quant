"""
Supply Chain Risk Analysis for Alternative Alpha
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from datetime import datetime, date
import logging

log = logging.getLogger(__name__)

class SupplyChainAnalyzer:
    """Analyze supply chain disruption risks and opportunities"""
    
    def __init__(self):
        # Mock supply chain relationships (in practice, from data providers)
        self.supply_chain_graph = self._build_mock_supply_chain()
        
    def _build_mock_supply_chain(self) -> Dict[str, Dict]:
        """Build mock supply chain relationships"""
        # In practice, this would come from supply chain data providers
        return {
            'AAPL': {
                'suppliers': ['TSM', 'QCOM', 'AVGO'],
                'customers': ['consumers'],
                'geographic_exposure': ['Asia', 'North America'],
                'key_materials': ['semiconductors', 'rare_earth'],
                'supply_chain_complexity': 0.8
            },
            'TSLA': {
                'suppliers': ['NVDA', 'PANW'],  # Mock suppliers
                'customers': ['consumers'],
                'geographic_exposure': ['Asia', 'North America', 'Europe'],
                'key_materials': ['lithium', 'semiconductors'],
                'supply_chain_complexity': 0.9
            },
            'TSM': {
                'suppliers': ['materials'],
                'customers': ['AAPL', 'NVDA', 'AMD'],
                'geographic_exposure': ['Asia'],
                'key_materials': ['silicon', 'rare_earth'],
                'supply_chain_complexity': 0.7
            }
        }
    
    def calculate_supply_chain_risk(self, symbols: List[str], 
                                  disruption_events: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Calculate supply chain risk scores for given symbols
        
        Args:
            symbols: List of symbols to analyze
            disruption_events: List of known disruption events
            
        Returns:
            DataFrame with supply chain risk metrics
        """
        if disruption_events is None:
            disruption_events = self._get_mock_disruption_events()
        
        risk_scores = []
        
        for symbol in symbols:
            if symbol not in self.supply_chain_graph:
                # For unknown symbols, assign neutral risk
                risk_scores.append({
                    'symbol': symbol,
                    'supply_chain_risk': 0.5,
                    'geographic_risk': 0.5,
                    'supplier_concentration': 0.5,
                    'disruption_exposure': 0.0,
                    'supply_chain_complexity': 0.5
                })
                continue
            
            company_data = self.supply_chain_graph[symbol]
            
            # Calculate risk components
            geographic_risk = self._calculate_geographic_risk(company_data['geographic_exposure'])
            supplier_risk = self._calculate_supplier_concentration_risk(company_data['suppliers'])
            disruption_risk = self._calculate_disruption_exposure(symbol, disruption_events)
            complexity_risk = company_data.get('supply_chain_complexity', 0.5)
            
            # Aggregate risk score
            overall_risk = (
                0.3 * geographic_risk +
                0.3 * supplier_risk +
                0.2 * disruption_risk +
                0.2 * complexity_risk
            )
            
            risk_scores.append({
                'symbol': symbol,
                'supply_chain_risk': overall_risk,
                'geographic_risk': geographic_risk,
                'supplier_concentration': supplier_risk,
                'disruption_exposure': disruption_risk,
                'supply_chain_complexity': complexity_risk
            })
        
        return pd.DataFrame(risk_scores)
    
    def _calculate_geographic_risk(self, geographic_exposure: List[str]) -> float:
        """Calculate risk based on geographic concentration"""
        # Risk scores by region (higher = more risky)
        region_risks = {
            'Asia': 0.7,          # Higher geopolitical risk
            'Europe': 0.4,        # Moderate risk
            'North America': 0.3, # Lower risk
            'Latin America': 0.6,
            'Africa': 0.8,
            'Middle East': 0.9
        }
        
        if not geographic_exposure:
            return 0.5
        
        # Calculate weighted average risk
        total_risk = sum(region_risks.get(region, 0.5) for region in geographic_exposure)
        return min(total_risk / len(geographic_exposure), 1.0)
    
    def _calculate_supplier_concentration_risk(self, suppliers: List[str]) -> float:
        """Calculate risk based on supplier concentration"""
        if not suppliers:
            return 0.5
        
        # More suppliers = lower concentration risk
        if len(suppliers) >= 5:
            return 0.2  # Well diversified
        elif len(suppliers) >= 3:
            return 0.4  # Moderately diversified
        elif len(suppliers) >= 2:
            return 0.6  # Some concentration
        else:
            return 0.9  # High concentration risk
    
    def _calculate_disruption_exposure(self, symbol: str, disruption_events: List[Dict]) -> float:
        """Calculate exposure to recent supply chain disruptions"""
        if not disruption_events:
            return 0.0
        
        company_data = self.supply_chain_graph.get(symbol, {})
        exposures = []
        
        for event in disruption_events:
            exposure = 0.0
            
            # Check geographic exposure
            if event.get('region') in company_data.get('geographic_exposure', []):
                exposure += 0.5
            
            # Check material/industry exposure
            if event.get('industry') in company_data.get('key_materials', []):
                exposure += 0.3
            
            # Check supplier exposure
            affected_companies = event.get('affected_companies', [])
            if any(supplier in affected_companies for supplier in company_data.get('suppliers', [])):
                exposure += 0.4
            
            exposures.append(min(exposure, 1.0))
        
        return np.mean(exposures) if exposures else 0.0
    
    def _get_mock_disruption_events(self) -> List[Dict]:
        """Get mock supply chain disruption events"""
        return [
            {
                'date': '2024-01-15',
                'region': 'Asia',
                'industry': 'semiconductors',
                'severity': 0.7,
                'affected_companies': ['TSM'],
                'description': 'Semiconductor factory shutdown'
            },
            {
                'date': '2024-02-01', 
                'region': 'Europe',
                'industry': 'energy',
                'severity': 0.6,
                'affected_companies': [],
                'description': 'Energy supply disruption'
            }
        ]
    
    def identify_supply_chain_alpha(self, price_data: pd.DataFrame, 
                                   risk_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Identify potential alpha opportunities from supply chain analysis
        
        Args:
            price_data: Historical price data
            risk_scores: Supply chain risk scores
            
        Returns:
            DataFrame with alpha signals
        """
        if price_data.empty or risk_scores.empty:
            return pd.DataFrame()
        
        # Merge data
        merged = price_data.merge(risk_scores, on='symbol', how='inner')
        
        alpha_signals = []
        
        for symbol, group in merged.groupby('symbol'):
            if len(group) < 20:  # Need sufficient history
                continue
            
            # Calculate price volatility
            returns = group['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Supply chain risk metrics
            risk_data = group.iloc[-1]  # Latest risk scores
            
            # Alpha signal generation
            supply_chain_alpha = self._generate_supply_chain_alpha(
                volatility, risk_data, returns
            )
            
            alpha_signals.append({
                'symbol': symbol,
                'supply_chain_alpha': supply_chain_alpha,
                'risk_adjusted_alpha': supply_chain_alpha / max(volatility, 0.1),
                'confidence': min(0.8, len(returns) / 252 * 0.5 + 0.3),
                'latest_risk_score': risk_data['supply_chain_risk']
            })
        
        return pd.DataFrame(alpha_signals)
    
    def _generate_supply_chain_alpha(self, volatility: float, 
                                   risk_data: pd.Series, 
                                   returns: pd.Series) -> float:
        """Generate alpha signal from supply chain analysis"""
        
        # High supply chain risk during stable periods = potential opportunity
        # Low supply chain risk during volatile periods = defensive play
        
        supply_chain_risk = risk_data['supply_chain_risk']
        recent_volatility = returns.tail(20).std() * np.sqrt(252)
        
        # Mean reversion signal based on risk-volatility mismatch
        volatility_percentile = (recent_volatility - returns.rolling(60).std().mean()) / returns.rolling(60).std().std()
        
        # Alpha signal
        if supply_chain_risk > 0.7 and volatility_percentile < -0.5:
            # High risk, low volatility = potential opportunity
            alpha = 0.3
        elif supply_chain_risk < 0.3 and volatility_percentile > 0.5:
            # Low risk, high volatility = defensive opportunity
            alpha = 0.2
        else:
            # Neutral or no clear signal
            alpha = 0.0
        
        # Adjust for disruption exposure
        disruption_exposure = risk_data.get('disruption_exposure', 0.0)
        if disruption_exposure > 0.5:
            alpha -= 0.1  # Reduce alpha for high disruption exposure
        
        return np.clip(alpha, -0.5, 0.5)