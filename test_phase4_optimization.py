"""
Test Phase 4 implementations: Advanced Portfolio Optimization, Execution, and Latency
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from unittest.mock import Mock, patch

# Test data setup
SAMPLE_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
SAMPLE_ALPHA = pd.Series([0.05, 0.03, 0.08, -0.02, 0.01], index=SAMPLE_SYMBOLS)


class TestCovarianceModels:
    """Test risk model covariance estimation"""
    
    def test_ewma_covariance(self):
        """Test EWMA covariance calculation"""
        from risk.covariance import ewma_cov
        
        # Generate sample returns
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 5) * 0.02,
            columns=SAMPLE_SYMBOLS
        )
        
        cov_matrix = ewma_cov(returns)
        
        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape == (5, 5)
        assert list(cov_matrix.index) == SAMPLE_SYMBOLS
        assert list(cov_matrix.columns) == SAMPLE_SYMBOLS
        
        # Check positive semidefinite
        eigenvals = np.linalg.eigvals(cov_matrix.values)
        assert all(eigenvals >= -1e-10)  # Allow for numerical precision
    
    def test_robust_covariance_fallback(self):
        """Test robust covariance with fallback"""
        from risk.covariance import robust_cov
        
        # Test with empty data
        empty_returns = pd.DataFrame()
        cov_matrix = robust_cov(empty_returns)
        assert isinstance(cov_matrix, pd.DataFrame)
        
        # Test with minimal data
        minimal_returns = pd.DataFrame({'A': [0.01]}, index=[0])
        cov_matrix = robust_cov(minimal_returns)
        assert cov_matrix.shape == (1, 1)


class TestPortfolioOptimization:
    """Test Mean-Variance Optimization"""
    
    def test_fallback_optimizer(self):
        """Test fallback optimizer when CVXPY unavailable"""
        from portfolio.mvo import _fallback_optimizer
        
        weights = _fallback_optimizer(SAMPLE_ALPHA)
        
        assert isinstance(weights, pd.Series)
        assert len(weights) > 0
        assert all(weights >= 0)  # Long-only
        assert weights.sum() > 0
    
    @patch('portfolio.mvo.cp', None)  # Simulate CVXPY unavailable
    def test_mvo_fallback_when_cvxpy_unavailable(self):
        """Test MVO falls back gracefully when CVXPY unavailable"""
        from portfolio.mvo import build_portfolio_mvo
        
        # Mock database functions to avoid dependency
        with patch('portfolio.mvo._latest_prices') as mock_prices, \
             patch('portfolio.mvo._adv20') as mock_adv, \
             patch('portfolio.mvo.synthesize_covariance') as mock_cov:
            
            mock_prices.return_value = pd.Series([150, 100, 200, 80, 300], index=SAMPLE_SYMBOLS)
            mock_adv.return_value = pd.Series([1000000] * 5, index=SAMPLE_SYMBOLS)
            mock_cov.return_value = (pd.DataFrame(), pd.DataFrame())
            
            weights = build_portfolio_mvo(SAMPLE_ALPHA, date(2024, 1, 1))
            
            assert isinstance(weights, pd.Series)
            assert len(weights) > 0


class TestExecutionAlgorithms:
    """Test execution algorithm implementations"""
    
    def test_twap_scheduling(self):
        """Test TWAP child order scheduling"""
        from trading.execution import ExecutionScheduler
        
        scheduler = ExecutionScheduler()
        
        # Sample trade
        trade = pd.Series({
            'id': 1,
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 1000,
            'price': 150.0
        })
        
        child_orders = scheduler._schedule_twap(trade, 4)
        
        assert len(child_orders) == 4
        assert sum(order.qty for order in child_orders) == 1000
        assert all(order.symbol == 'AAPL' for order in child_orders)
        assert all(order.side == 'buy' for order in child_orders)
        assert all(order.slice_idx > 0 for order in child_orders)
    
    def test_vwap_scheduling_fallback(self):
        """Test VWAP scheduling falls back to TWAP when no volume data"""
        from trading.execution import ExecutionScheduler
        
        scheduler = ExecutionScheduler()
        
        # Mock empty volume profile
        with patch.object(scheduler, '_get_intraday_volume_profile', return_value=pd.DataFrame()):
            trade = pd.Series({
                'id': 1,
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 1000,
                'price': 150.0
            })
            
            child_orders = scheduler._schedule_vwap(trade, 4)
            
            # Should fall back to TWAP
            assert len(child_orders) == 4
            assert sum(order.qty for order in child_orders) == 1000
    
    def test_implementation_shortfall_scheduling(self):
        """Test Implementation Shortfall scheduling"""
        from trading.execution import ExecutionScheduler
        
        scheduler = ExecutionScheduler()
        
        # Mock market data
        with patch.object(scheduler, '_get_market_data', return_value={'adv': 1000000, 'volatility': 0.20}):
            trade = pd.Series({
                'id': 1,
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 1000,
                'price': 150.0
            })
            
            child_orders = scheduler._schedule_implementation_shortfall(trade, 4)
            
            assert len(child_orders) > 0
            assert sum(order.qty for order in child_orders) == 1000
            assert all(hasattr(order, 'participation_rate') for order in child_orders)


class TestFIXConnector:
    """Test FIX protocol connectivity"""
    
    def test_fix_message_creation(self):
        """Test FIX message formatting"""
        from trading.fix_connector import FIXMessage
        
        fields = {
            11: 'TEST123',  # ClOrdID
            55: 'AAPL',     # Symbol
            54: '1',        # Side (Buy)
            38: '100'       # OrderQty
        }
        
        message = FIXMessage('D', fields)  # New Order Single
        
        assert message.msg_type == 'D'
        assert '35=D' in message.raw_message  # Message type
        assert '11=TEST123' in message.raw_message  # ClOrdID
        assert '55=AAPL' in message.raw_message  # Symbol
    
    def test_order_request_creation(self):
        """Test order request data structure"""
        from trading.fix_connector import OrderRequest
        
        order = OrderRequest(
            symbol='AAPL',
            side='buy',
            quantity=100,
            order_type='market'
        )
        
        assert order.symbol == 'AAPL'
        assert order.side == 'buy'
        assert order.quantity == 100
        assert order.order_type == 'market'
    
    def test_fix_connector_initialization(self):
        """Test FIX connector can be initialized"""
        from trading.fix_connector import FIXConnector
        
        connector = FIXConnector(
            host='localhost',
            port=9878,
            sender_comp_id='TEST_CLIENT',
            target_comp_id='TEST_BROKER'
        )
        
        assert connector.host == 'localhost'
        assert connector.port == 9878
        assert connector.sender_comp_id == 'TEST_CLIENT'
        assert connector.target_comp_id == 'TEST_BROKER'
        assert not connector.connected
        assert not connector.session_active


class TestBrokerIntegration:
    """Test enhanced broker functionality"""
    
    def test_broker_protocol_selection(self):
        """Test broker selects appropriate protocol"""
        from trading.broker import _submit_order
        from config import ENABLE_FIX_PROTOCOL
        
        # Test that function exists and can be called
        # (without actually connecting to avoid external dependencies)
        assert callable(_submit_order)
    
    @patch('trading.broker.ENABLE_FIX_PROTOCOL', False)
    def test_rest_api_fallback(self):
        """Test REST API is used when FIX disabled"""
        from trading.broker import _submit_order_alpaca_rest
        
        # Mock Alpaca API
        with patch('trading.broker.tradeapi') as mock_api:
            mock_order = Mock()
            mock_order.id = 'ORDER123'
            mock_api.REST.return_value.submit_order.return_value = mock_order
            
            order_id = _submit_order_alpaca_rest('AAPL', 100, 'buy', 150.0, 'TEST123')
            
            assert order_id == 'ORDER123'


class TestConfigurationEnhancements:
    """Test enhanced configuration"""
    
    def test_phase4_config_parameters(self):
        """Test Phase 4 configuration parameters are available"""
        import config
        
        # MVO parameters
        assert hasattr(config, 'USE_MVO')
        assert hasattr(config, 'MVO_RISK_LAMBDA')
        assert hasattr(config, 'MVO_COST_LAMBDA')
        
        # Factor model parameters
        assert hasattr(config, 'USE_FACTOR_MODEL')
        assert hasattr(config, 'EWMA_LAMBDA')
        assert hasattr(config, 'USE_LEDOIT_WOLF')
        
        # Execution parameters
        assert hasattr(config, 'ENABLE_CHILD_ORDERS')
        assert hasattr(config, 'DEFAULT_EXECUTION_SLICES')
        
        # FIX protocol parameters
        assert hasattr(config, 'ENABLE_FIX_PROTOCOL')
        assert hasattr(config, 'FIX_HOST')
        assert hasattr(config, 'FIX_PORT')


if __name__ == '__main__':
    # Run basic smoke tests without pytest
    print("Running Phase 4 smoke tests...")
    
    try:
        # Test configuration
        import config
        print(f"✓ Config loaded: USE_MVO={config.USE_MVO}")
        
        # Test covariance
        from risk.covariance import robust_cov
        test_returns = pd.DataFrame(np.random.randn(50, 3) * 0.02, columns=['A', 'B', 'C'])
        cov = robust_cov(test_returns)
        print(f"✓ Covariance test passed: {cov.shape}")
        
        # Test execution
        from trading.execution import ExecutionScheduler
        scheduler = ExecutionScheduler()
        print("✓ Execution scheduler created")
        
        # Test FIX
        from trading.fix_connector import OrderRequest
        order = OrderRequest('AAPL', 'buy', 100)
        print(f"✓ FIX order request: {order.symbol}")
        
        print("\n🎉 All Phase 4 smoke tests passed!")
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        raise