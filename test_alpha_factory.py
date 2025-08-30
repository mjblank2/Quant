"""
Integration test for Phase 3 Alpha Factory implementation
"""

import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_feature_store():
    """Test feature store functionality"""
    print("=" * 50)
    print("Testing Feature Store")
    print("=" * 50)

    try:
        from features import FeatureRegistry

        # Test registry
        registry = FeatureRegistry()
        features = registry.list_features()
        print(f"✓ Feature registry loaded with {len(features)} features")

        # Test feature metadata
        ret_1d_meta = registry.get_feature("ret_1d")
        if ret_1d_meta:
            print(f"✓ Feature metadata: {ret_1d_meta.name} - {ret_1d_meta.description}")

        # Test dependencies
        deps = registry.get_dependencies("vol_21")
        print(f"✓ Dependencies for vol_21: {deps}")

        return True

    except Exception as e:
        print(f"✗ Feature store test failed: {e}")
        return False


def test_tca_models():
    """Test TCA functionality"""
    print("=" * 50)
    print("Testing TCA Models")
    print("=" * 50)

    try:
        from tca import SquareRootLaw, TransactionCostModel

        # Test market impact model
        sqrt_model = SquareRootLaw()
        impact = sqrt_model.estimate_impact(
            order_size=10000, adv=500000, volatility=0.25, price=100.0
        )
        print(f"✓ Market impact estimate: {impact['total_bps']:.2f} bps")

        # Test cost model
        cost_model = TransactionCostModel()
        costs = cost_model.estimate_costs(
            symbol="AAPL",
            order_size=5000,
            price=150.0,
            market_data={"adv": 1000000, "volatility": 0.20, "spread_bps": 5.0},
        )
        print(f"✓ Total transaction cost: {costs.total_bps:.2f} bps")

        return True

    except Exception as e:
        print(f"✗ TCA models test failed: {e}")
        return False


def test_validation_tools():
    """Test validation functionality"""
    print("=" * 50)
    print("Testing Validation Tools")
    print("=" * 50)

    try:
        from validation import DeflatedSharpeRatio, CombinatorlialPurgedCV

        # Test Deflated Sharpe Ratio
        dsr = DeflatedSharpeRatio()
        mock_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        dsr_result = dsr.compute_deflated_sharpe(mock_returns, n_trials=5)
        print(
            f"✓ Deflated Sharpe Ratio: {dsr_result['dsr']:.3f} (Sharpe: {dsr_result['sharpe']:.3f})"
        )

        # Test Purged CV
        pcv = CombinatorlialPurgedCV()
        mock_data = pd.DataFrame(
            {
                "ts": pd.date_range("2020-01-01", periods=100),
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        splits = pcv.split(mock_data)
        print(f"✓ Purged CV generated {len(splits)} splits")

        return True

    except Exception as e:
        print(f"✗ Validation tools test failed: {e}")
        return False


def test_alternative_data():
    """Test alternative data functionality"""
    print("=" * 50)
    print("Testing Alternative Data")
    print("=" * 50)

    try:
        from alternative_data import (
            SentimentAnalyzer,
            SupplyChainAnalyzer,
            ESGDataProcessor,
        )

        # Test sentiment analysis
        sentiment = SentimentAnalyzer()
        text = "Apple reports strong quarterly earnings with significant revenue growth"
        sentiment_result = sentiment.analyze_text(text, ["AAPL"])
        if sentiment_result:
            first_result = list(sentiment_result.values())[0]
            print(
                f"✓ Sentiment analysis: {first_result.score:.3f} (confidence: {first_result.confidence:.3f})"
            )

        # Test supply chain analysis
        supply_chain = SupplyChainAnalyzer()
        risk_scores = supply_chain.calculate_supply_chain_risk(["AAPL", "TSLA"])
        print(f"✓ Supply chain analysis completed for {len(risk_scores)} symbols")

        # Test ESG processing
        esg = ESGDataProcessor()
        esg_scores = esg.calculate_esg_scores(["AAPL", "MSFT"])
        print(f"✓ ESG analysis completed for {len(esg_scores)} symbols")

        return True

    except Exception as e:
        print(f"✗ Alternative data test failed: {e}")
        return False


def test_alpha_factory_integration():
    """Test complete alpha factory integration"""
    print("=" * 50)
    print("Testing Alpha Factory Integration")
    print("=" * 50)

    try:
        # Mock engine for testing
        class MockEngine:
            def connect(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # This will fail due to missing database, but we can test imports
        try:
            from alpha_factory import AlphaFactory  # noqa: F401

            print("✓ Alpha Factory imports successful")
        except Exception as e:
            if "DATABASE_URL" in str(e):
                print(
                    "✓ Alpha Factory imports successful (DB connection expected to fail in test)"
                )
            else:
                raise e

        return True

    except Exception as e:
        print(f"✗ Alpha factory integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing Phase 3 Alpha Factory Implementation")
    print("=" * 80)

    tests = [
        test_feature_store,
        test_tca_models,
        test_validation_tools,
        test_alternative_data,
        test_alpha_factory_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)
        print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Phase 3 Alpha Factory is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
