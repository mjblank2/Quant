"""
Alpha Factory Integration - Ties together all Phase 3 components
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import logging

# Import Phase 3 components
from features import FeatureStore, FeatureRegistry
from tca import MarketImpactModel, SquareRootLaw, ExecutionAnalyzer, TransactionCostModel
from validation import DeflatedSharpeRatio, CombinatorlialPurgedCV, WalkForwardAnalysis
from alternative_data import SentimentAnalyzer, SupplyChainAnalyzer, ESGDataProcessor

log = logging.getLogger(__name__)

class AlphaFactory:
    """
    Industrialized Alpha Research Factory
    Integrates feature store, sophisticated TCA, statistical validation, and alternative data
    """
    
    def __init__(self, engine):
        # Core components
        self.engine = engine
        self.feature_store = FeatureStore(engine)
        self.feature_registry = FeatureRegistry()
        
        # TCA components
        self.market_impact_model = SquareRootLaw()
        self.execution_analyzer = ExecutionAnalyzer(self.market_impact_model)
        self.cost_model = TransactionCostModel(market_impact_model=self.market_impact_model)
        
        # Validation components
        self.deflated_sharpe = DeflatedSharpeRatio()
        self.purged_cv = CombinatorlialPurgedCV()
        self.walk_forward = WalkForwardAnalysis()
        
        # Alternative data components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.supply_chain_analyzer = SupplyChainAnalyzer()
        self.esg_processor = ESGDataProcessor()
        
        log.info("Alpha Factory initialized with all Phase 3 components")
    
    def run_full_alpha_research_pipeline(self, symbols: List[str], 
                                       start_date: Optional[date] = None,
                                       end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Run complete alpha research pipeline
        
        Args:
            symbols: List of symbols to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Comprehensive alpha research results
        """
        log.info(f"Starting alpha research pipeline for {len(symbols)} symbols")
        
        results = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'pipeline_timestamp': datetime.now()
        }
        
        try:
            # 1. Feature Engineering with Consistency
            log.info("Step 1: Feature engineering with centralized store")
            features_result = self._run_feature_engineering(symbols, start_date, end_date)
            results['features'] = features_result
            
            # 2. Alternative Data Integration
            log.info("Step 2: Alternative data integration")
            alt_data_result = self._integrate_alternative_data(symbols)
            results['alternative_data'] = alt_data_result
            
            # 3. Advanced Model Validation
            log.info("Step 3: Advanced statistical validation")
            validation_result = self._run_advanced_validation(features_result['features'])
            results['validation'] = validation_result
            
            # 4. Sophisticated TCA Analysis
            log.info("Step 4: Transaction cost analysis")
            tca_result = self._run_tca_analysis(symbols, features_result['features'])
            results['tca'] = tca_result
            
            # 5. Alpha Signal Generation
            log.info("Step 5: Alpha signal generation")
            alpha_result = self._generate_alpha_signals(
                features_result['features'], 
                alt_data_result,
                validation_result
            )
            results['alpha_signals'] = alpha_result
            
            # 6. Performance Attribution
            log.info("Step 6: Performance attribution and reporting")
            attribution_result = self._run_performance_attribution(results)
            results['attribution'] = attribution_result
            
            results['status'] = 'success'
            log.info("Alpha research pipeline completed successfully")
            
        except Exception as e:
            log.error(f"Alpha research pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _run_feature_engineering(self, symbols: List[str], 
                               start_date: Optional[date], 
                               end_date: Optional[date]) -> Dict[str, Any]:
        """Run centralized feature engineering"""
        
        # Compute features using centralized store
        features_df = self.feature_store.compute_features(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get latest features for serving
        latest_features = self.feature_store.get_latest_features(symbols)
        
        # Validate training/serving consistency
        consistency_check = {}
        if not features_df.empty and not latest_features.empty:
            consistency_check = self.feature_store.validate_feature_consistency(
                features_df.tail(100),  # Last 100 training observations
                latest_features
            )
        
        # Feature metadata
        feature_metadata = {}
        for feature_name in features_df.columns:
            if feature_name not in ['symbol', 'ts']:
                feature_metadata[feature_name] = self.feature_store.get_feature_metadata(feature_name)
        
        return {
            'features': features_df,
            'latest_features': latest_features,
            'consistency_check': consistency_check,
            'feature_metadata': feature_metadata,
            'n_features': len([col for col in features_df.columns if col not in ['symbol', 'ts']]),
            'n_observations': len(features_df)
        }
    
    def _integrate_alternative_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Integrate all alternative data sources"""
        
        alt_data_results = {}
        
        try:
            # Sentiment analysis
            from alternative_data.sentiment import integrate_sentiment_with_features
            sentiment_features = integrate_sentiment_with_features(
                self.feature_store, symbols, self.sentiment_analyzer
            )
            alt_data_results['sentiment'] = sentiment_features
            
        except Exception as e:
            log.warning(f"Sentiment analysis failed: {e}")
            alt_data_results['sentiment'] = pd.DataFrame()
        
        try:
            # Supply chain analysis
            supply_chain_risk = self.supply_chain_analyzer.calculate_supply_chain_risk(symbols)
            alt_data_results['supply_chain'] = supply_chain_risk
            
        except Exception as e:
            log.warning(f"Supply chain analysis failed: {e}")
            alt_data_results['supply_chain'] = pd.DataFrame()
        
        try:
            # ESG analysis
            esg_scores = self.esg_processor.calculate_esg_scores(symbols)
            esg_alpha = self.esg_processor.generate_esg_alpha_signals(
                esg_scores, pd.DataFrame()  # Would need price data in practice
            )
            alt_data_results['esg'] = {
                'scores': esg_scores,
                'alpha_signals': esg_alpha
            }
            
        except Exception as e:
            log.warning(f"ESG analysis failed: {e}")
            alt_data_results['esg'] = {'scores': pd.DataFrame(), 'alpha_signals': pd.DataFrame()}
        
        # Summary statistics
        alt_data_summary = {
            'sentiment_coverage': len(alt_data_results.get('sentiment', pd.DataFrame())),
            'supply_chain_coverage': len(alt_data_results.get('supply_chain', pd.DataFrame())),
            'esg_coverage': len(alt_data_results.get('esg', {}).get('scores', pd.DataFrame())),
            'total_alt_features': sum([
                len(df) for df in [
                    alt_data_results.get('sentiment', pd.DataFrame()),
                    alt_data_results.get('supply_chain', pd.DataFrame()),
                    alt_data_results.get('esg', {}).get('scores', pd.DataFrame())
                ]
            ])
        }
        
        return {
            'data': alt_data_results,
            'summary': alt_data_summary
        }
    
    def _run_advanced_validation(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Run advanced statistical validation"""
        
        if features_df.empty:
            return {'error': 'No features available for validation'}
        
        validation_results = {}
        
        try:
            # Combinatorial Purged Cross-Validation
            cv_splits = self.purged_cv.split(features_df)
            cv_validation = self.purged_cv.validate_splits(cv_splits, features_df['ts'])
            validation_results['purged_cv'] = {
                'splits': len(cv_splits),
                'validation': cv_validation
            }
            
        except Exception as e:
            log.warning(f"Purged CV failed: {e}")
            validation_results['purged_cv'] = {'error': str(e)}
        
        try:
            # Walk-Forward Analysis (simplified)
            if 'ts' in features_df.columns and len(features_df) > 500:
                features_df_indexed = features_df.set_index('ts')
                
                # Mock model function for walk-forward
                def mock_model_func(X_train, y_train, X_test):
                    return np.random.normal(0, 0.1, len(X_test))
                
                # Add mock target for demonstration
                features_df_indexed['target'] = np.random.normal(0, 0.1, len(features_df_indexed))
                
                wf_results = self.walk_forward.run_analysis(
                    features_df_indexed, mock_model_func, 'target'
                )
                validation_results['walk_forward'] = wf_results
                
        except Exception as e:
            log.warning(f"Walk-forward analysis failed: {e}")
            validation_results['walk_forward'] = {'error': str(e)}
        
        try:
            # Deflated Sharpe Ratio calculation
            mock_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Mock daily returns
            dsr_result = self.deflated_sharpe.compute_deflated_sharpe(
                mock_returns, n_trials=10
            )
            validation_results['deflated_sharpe'] = dsr_result
            
        except Exception as e:
            log.warning(f"Deflated Sharpe calculation failed: {e}")
            validation_results['deflated_sharpe'] = {'error': str(e)}
        
        return validation_results
    
    def _run_tca_analysis(self, symbols: List[str], features_df: pd.DataFrame) -> Dict[str, Any]:
        """Run sophisticated transaction cost analysis"""
        
        tca_results = {}
        
        try:
            # Mock trade data for TCA demonstration
            mock_trades = pd.DataFrame({
                'symbol': symbols[:min(5, len(symbols))],  # Limit for demo
                'quantity': [1000, -500, 2000, -1500, 800],
                'price': [100.0, 50.0, 200.0, 75.0, 150.0],
                'execution_style': ['twap', 'vwap', 'market', 'twap', 'vwap']
            })
            
            # Mock market data
            mock_market_data = pd.DataFrame({
                'symbol': mock_trades['symbol'],
                'adv': [500000, 200000, 1000000, 300000, 600000],
                'volatility': [0.25, 0.30, 0.20, 0.35, 0.22],
                'spread_bps': [8.0, 12.0, 6.0, 15.0, 9.0]
            })
            
            # Estimate portfolio costs
            cost_estimates = self.cost_model.estimate_portfolio_costs(mock_trades, mock_market_data)
            tca_results['cost_estimates'] = cost_estimates.to_dict('records') if not cost_estimates.empty else []
            
            # Benchmark execution styles for first symbol
            if not mock_trades.empty:
                first_symbol = mock_trades.iloc[0]['symbol']
                first_market_data = mock_market_data.iloc[0].to_dict()
                
                style_comparison = self.cost_model.benchmark_execution_styles(
                    first_symbol, 1000, 100.0, first_market_data
                )
                tca_results['execution_style_comparison'] = style_comparison.to_dict('records')
            
            # TCA summary
            if not cost_estimates.empty:
                tca_summary = {
                    'total_estimated_cost_bps': cost_estimates['total_cost_bps'].mean(),
                    'total_estimated_cost_usd': cost_estimates['total_cost_usd'].sum(),
                    'avg_market_impact_bps': cost_estimates['market_impact_bps'].mean(),
                    'n_trades_analyzed': len(cost_estimates)
                }
                tca_results['summary'] = tca_summary
            
        except Exception as e:
            log.warning(f"TCA analysis failed: {e}")
            tca_results['error'] = str(e)
        
        return tca_results
    
    def _generate_alpha_signals(self, features_df: pd.DataFrame, 
                              alt_data_result: Dict[str, Any],
                              validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive alpha signals"""
        
        alpha_signals = {}
        
        try:
            # Combine traditional and alternative features
            combined_features = features_df.copy()
            
            # Add alternative data features
            alt_data = alt_data_result.get('data', {})
            
            # Sentiment features
            sentiment_df = alt_data.get('sentiment', pd.DataFrame())
            if not sentiment_df.empty and 'symbol' in sentiment_df.columns:
                combined_features = combined_features.merge(
                    sentiment_df, on='symbol', how='left'
                )
            
            # Supply chain features
            supply_chain_df = alt_data.get('supply_chain', pd.DataFrame())
            if not supply_chain_df.empty and 'symbol' in supply_chain_df.columns:
                combined_features = combined_features.merge(
                    supply_chain_df, on='symbol', how='left'
                )
            
            # ESG features
            esg_data = alt_data.get('esg', {})
            esg_alpha_df = esg_data.get('alpha_signals', pd.DataFrame())
            if not esg_alpha_df.empty and 'symbol' in esg_alpha_df.columns:
                combined_features = combined_features.merge(
                    esg_alpha_df, on='symbol', how='left'
                )
            
            # Generate meta-signals
            alpha_signals['feature_count'] = len([col for col in combined_features.columns if col not in ['symbol', 'ts']])
            alpha_signals['symbols_with_features'] = combined_features['symbol'].nunique()
            
            # Simple alpha score (in practice, would use sophisticated models)
            if not combined_features.empty:
                numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['ts']]
                
                if len(numeric_cols) > 0:
                    # Normalize features and create simple composite score
                    feature_matrix = combined_features[numeric_cols].fillna(0)
                    normalized_features = (feature_matrix - feature_matrix.mean()) / (feature_matrix.std() + 1e-8)
                    
                    # Simple equal-weight combination
                    alpha_score = normalized_features.mean(axis=1)
                    combined_features['alpha_score'] = alpha_score
                    
                    alpha_signals['alpha_scores'] = combined_features[['symbol', 'alpha_score']].to_dict('records')
            
        except Exception as e:
            log.warning(f"Alpha signal generation failed: {e}")
            alpha_signals['error'] = str(e)
        
        return alpha_signals
    
    def _run_performance_attribution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance attribution analysis"""
        
        attribution = {
            'pipeline_summary': {
                'total_symbols': len(results.get('symbols', [])),
                'features_generated': results.get('features', {}).get('n_features', 0),
                'alt_data_sources': len([k for k, v in results.get('alternative_data', {}).get('data', {}).items() if (not v.empty if isinstance(v, pd.DataFrame) else True)]),
                'validation_tests': len([k for k, v in results.get('validation', {}).items() if 'error' not in v]),
                'tca_trades_analyzed': results.get('tca', {}).get('summary', {}).get('n_trades_analyzed', 0)
            }
        }
        
        # Feature importance (mock)
        feature_importance = {
            'traditional_features': 0.6,
            'alternative_data': 0.25,
            'market_microstructure': 0.15
        }
        attribution['feature_importance'] = feature_importance
        
        # Quality metrics
        quality_metrics = {
            'feature_consistency': results.get('features', {}).get('consistency_check', {}).get('consistent', False),
            'validation_passed': len([k for k, v in results.get('validation', {}).items() if 'error' not in v]) > 0,
            'tca_coverage': results.get('tca', {}).get('summary', {}).get('n_trades_analyzed', 0) > 0
        }
        attribution['quality_metrics'] = quality_metrics
        
        return attribution
    
    def get_factory_status(self) -> Dict[str, Any]:
        """Get status of alpha factory components"""
        return {
            'feature_store': {
                'registry_features': len(self.feature_registry._features),
                'available': True
            },
            'tca_models': {
                'market_impact_model': type(self.market_impact_model).__name__,
                'available': True
            },
            'validation_tools': {
                'deflated_sharpe': True,
                'purged_cv': True,
                'walk_forward': True
            },
            'alternative_data': {
                'sentiment': True,
                'supply_chain': True,
                'esg': True
            },
            'factory_ready': True
        }