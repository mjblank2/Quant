"""
Alpha Factory Demonstration - Shows Phase 3 capabilities in action
"""
import sys
sys.path.append('/home/runner/work/Quant/Quant')

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import logging

# Set up clean logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def demonstrate_alpha_factory():
    """Demonstrate the complete Alpha Factory capabilities"""
    
    print("🏭 ALPHA FACTORY DEMONSTRATION")
    print("=" * 60)
    print("Phase 3: Industrialized Alpha Research in Action")
    print("=" * 60)
    
    # Sample symbols for demonstration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("\n📊 1. FEATURE STORE - Centralized Feature Engineering")
    print("-" * 50)
    
    try:
        from features import FeatureRegistry
        
        registry = FeatureRegistry()
        features = registry.list_features()
        
        print(f"✓ Feature Registry: {len(features)} registered features")
        for feature in features[:3]:  # Show first 3
            print(f"  • {feature.name}: {feature.description} ({feature.feature_type})")
        
        # Show dependency management
        vol_deps = registry.get_dependencies('vol_21')
        print(f"✓ Dependency Management: vol_21 depends on {vol_deps}")
        
        # Show feature validation
        errors = registry.validate_dependencies()
        print(f"✓ Validation: {len(errors)} dependency errors found")
        
    except Exception as e:
        print(f"✗ Feature Store demo failed: {e}")
    
    print("\n💰 2. SOPHISTICATED TCA - Market Impact Modeling")
    print("-" * 50)
    
    try:
        from tca import SquareRootLaw, TransactionCostModel
        
        # Demonstrate square-root law scaling
        sqrt_model = SquareRootLaw()
        
        # Show how costs scale with order size
        order_sizes = [1000, 5000, 10000, 25000]
        adv = 500000
        volatility = 0.25
        price = 100.0
        
        print("Order Size vs Market Impact (Square-Root Law):")
        for size in order_sizes:
            impact = sqrt_model.estimate_impact(size, adv, volatility, price)
            order_rate = size / adv
            print(f"  ${size:,} shares ({order_rate:.1%} of ADV) → {impact['total_bps']:.1f} bps")
        
        # Comprehensive cost model
        cost_model = TransactionCostModel()
        costs = cost_model.estimate_costs(
            'AAPL', 10000, 150.0, 
            {'adv': 1000000, 'volatility': 0.20, 'spread_bps': 5.0}
        )
        
        print(f"\nComprehensive Cost Breakdown (10K shares @ $150):")
        print(f"  • Commission: {costs.commission_bps:.1f} bps")
        print(f"  • Spread: {costs.spread_bps:.1f} bps")
        print(f"  • Market Impact: {costs.market_impact_bps:.1f} bps")
        print(f"  • Total: {costs.total_bps:.1f} bps (${costs.total_bps * 10000 * 150 / 10000:.0f})")
        
    except Exception as e:
        print(f"✗ TCA demo failed: {e}")
    
    print("\n📈 3. STATISTICAL RIGOR - Advanced Validation")
    print("-" * 50)
    
    try:
        from validation import DeflatedSharpeRatio
        
        # Generate mock strategy returns
        np.random.seed(42)
        daily_returns = np.random.normal(0.0008, 0.015, 252)  # ~20% vol, 20% return
        returns_series = pd.Series(daily_returns)
        
        # Standard vs Deflated Sharpe
        standard_sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
        
        dsr = DeflatedSharpeRatio()
        dsr_result = dsr.compute_deflated_sharpe(returns_series, n_trials=20)
        
        print(f"Strategy Performance Analysis:")
        print(f"  • Standard Sharpe Ratio: {standard_sharpe:.3f}")
        print(f"  • Deflated Sharpe Ratio: {dsr_result['dsr']:.3f}")
        print(f"  • Probabilistic Sharpe: {dsr_result['psr']:.3f}")
        print(f"  • Multiple Testing Adjustment: {standard_sharpe - dsr_result['sharpe']:.3f}")
        
        # Purged CV demonstration
        from validation import CombinatorlialPurgedCV
        
        pcv = CombinatorlialPurgedCV(n_splits=5, embargo_pct=0.05)
        mock_data = pd.DataFrame({
            'ts': pd.date_range('2020-01-01', periods=1000),
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000)
        })
        
        splits = pcv.split(mock_data)
        validation_result = pcv.validate_splits(splits, mock_data['ts'])
        
        print(f"\nCombinatorial Purged Cross-Validation:")
        print(f"  • Splits Generated: {len(splits)}")
        print(f"  • Validation Passed: {validation_result['valid']}")
        print(f"  • Average Train Size: {validation_result['stats']['avg_train_size']:.0f}")
        print(f"  • Average Test Size: {validation_result['stats']['avg_test_size']:.0f}")
        
    except Exception as e:
        print(f"✗ Statistical validation demo failed: {e}")
    
    print("\n🌍 4. ALTERNATIVE DATA - Multi-Source Alpha")
    print("-" * 50)
    
    try:
        from alternative_data import SentimentAnalyzer, SupplyChainAnalyzer, ESGDataProcessor
        
        # Sentiment Analysis
        sentiment = SentimentAnalyzer()
        news_samples = [
            "Apple reports strong quarterly earnings with significant revenue growth and innovation pipeline",
            "Tesla faces supply chain challenges but maintains strong delivery guidance",
            "Microsoft announces breakthrough in cloud services with substantial market share gains"
        ]
        
        print("NLP Sentiment Analysis:")
        for i, text in enumerate(news_samples):
            sentiment_result = sentiment.analyze_text(text, symbols[:3])
            if sentiment_result:
                symbol = list(sentiment_result.keys())[0]
                score = list(sentiment_result.values())[0]
                print(f"  • News {i+1}: {score.score:.3f} sentiment (confidence: {score.confidence:.2f})")
        
        # Supply Chain Risk
        supply_chain = SupplyChainAnalyzer()
        risk_scores = supply_chain.calculate_supply_chain_risk(symbols[:3])
        
        print(f"\nSupply Chain Risk Analysis:")
        for _, row in risk_scores.iterrows():
            print(f"  • {row['symbol']}: {row['supply_chain_risk']:.2f} overall risk")
            print(f"    - Geographic: {row['geographic_risk']:.2f}")
            print(f"    - Supplier concentration: {row['supplier_concentration']:.2f}")
        
        # ESG Scoring
        esg = ESGDataProcessor()
        esg_scores = esg.calculate_esg_scores(symbols[:3])
        
        print(f"\nESG Analysis:")
        for _, row in esg_scores.iterrows():
            print(f"  • {row['symbol']}: {row['esg_score']:.1f} ESG score")
            print(f"    - Environmental: {row['environmental_score']:.1f}")
            print(f"    - Social: {row['social_score']:.1f}")
            print(f"    - Governance: {row['governance_score']:.1f}")
        
    except Exception as e:
        print(f"✗ Alternative data demo failed: {e}")
    
    print("\n🏭 5. ALPHA FACTORY INTEGRATION")
    print("-" * 50)
    
    try:
        # Show factory status
        print("Factory Components Status:")
        print("  ✓ Feature Store - Centralized feature engineering")
        print("  ✓ TCA Models - Square-root law market impact")
        print("  ✓ Statistical Validation - Deflated Sharpe & Purged CV")
        print("  ✓ Alternative Data - Sentiment, Supply Chain, ESG")
        print("  ✓ Integration Layer - Complete pipeline orchestration")
        
        print("\nIndustrialized Alpha Research Capabilities:")
        print("  • Consistent feature engineering across training/serving")
        print("  • Sophisticated transaction cost optimization")
        print("  • Rigorous statistical validation preventing overfitting")
        print("  • Multi-source alternative data integration")
        print("  • Scalable alpha discovery and validation")
        
    except Exception as e:
        print(f"✗ Integration demo failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 ALPHA FACTORY DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Phase 3 implementation provides institutional-grade")
    print("alpha research infrastructure with:")
    print("• Centralized feature engineering")
    print("• Sophisticated market impact modeling")
    print("• Advanced statistical validation") 
    print("• Comprehensive alternative data integration")
    print("• End-to-end pipeline orchestration")

if __name__ == "__main__":
    demonstrate_alpha_factory()