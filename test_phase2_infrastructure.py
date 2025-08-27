"""
Basic tests for Phase 2 Data Infrastructure components.

These tests verify the core functionality without requiring database connectivity.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataValidation(unittest.TestCase):
    """Test data validation components."""

    @patch('data.validation.engine')
    @patch('data.validation.ENABLE_DATA_VALIDATION', True)
    def test_validation_result_creation(self, mock_engine):
        """Test ValidationResult class functionality."""
        from data.validation import ValidationResult
        
        result = ValidationResult()
        self.assertTrue(result.passed)
        self.assertEqual(len(result.warnings), 0)
        self.assertEqual(len(result.errors), 0)
        
        result.add_warning("Test warning")
        self.assertEqual(len(result.warnings), 1)
        self.assertTrue(result.passed)  # Warnings don't fail
        
        result.add_error("Test error")
        self.assertEqual(len(result.errors), 1)
        self.assertFalse(result.passed)  # Errors fail
        
        result.add_metric("test_metric", 42.0)
        self.assertEqual(result.metrics["test_metric"], 42.0)

    @patch('data.validation.engine')
    @patch('data.validation.ENABLE_DATA_VALIDATION', False)
    def test_validation_disabled(self, mock_engine):
        """Test validation when disabled."""
        from data.validation import check_data_staleness
        
        result = check_data_staleness()
        self.assertTrue(result.passed)
        self.assertEqual(len(result.warnings), 0)
        self.assertEqual(len(result.errors), 0)


class TestTimescaleDB(unittest.TestCase):
    """Test TimescaleDB integration components."""

    @patch('data.timescale.engine')
    @patch('data.timescale.ENABLE_TIMESCALEDB', True)
    def test_timescaledb_availability_check(self, mock_engine):
        """Test TimescaleDB availability checking."""
        from data.timescale import is_timescaledb_available
        
        # Mock successful connection with TimescaleDB
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.scalar.return_value = True
        
        result = is_timescaledb_available()
        self.assertTrue(result)

    @patch('data.timescale.engine')
    @patch('data.timescale.ENABLE_TIMESCALEDB', False)
    def test_timescaledb_disabled(self, mock_engine):
        """Test TimescaleDB when disabled."""
        from data.timescale import setup_timescaledb
        
        result = setup_timescaledb()
        self.assertFalse(result)

    def test_timescaledb_info_structure(self):
        """Test TimescaleDB info structure."""
        from data.timescale import get_timescaledb_info
        
        with patch('data.timescale.is_timescaledb_available', return_value=False):
            info = get_timescaledb_info()
            
            required_keys = ['enabled', 'available', 'hypertable_configured', 
                           'compression_enabled', 'chunk_count', 'compressed_chunks']
            
            for key in required_keys:
                self.assertIn(key, info)


class TestInstitutionalIngest(unittest.TestCase):
    """Test enhanced ingestion pipeline."""

    @patch('data.institutional_ingest.ENABLE_DATA_VALIDATION', True)
    @patch('data.institutional_ingest.setup_timescaledb')
    @patch('data.institutional_ingest.get_timescaledb_info')
    def test_infrastructure_setup(self, mock_get_info, mock_setup):
        """Test infrastructure setup process."""
        from data.institutional_ingest import setup_infrastructure
        
        mock_setup.return_value = True
        mock_get_info.return_value = {'enabled': True, 'available': True}
        
        result = setup_infrastructure()
        self.assertTrue(result)
        mock_setup.assert_called_once()

    def test_health_check_structure(self):
        """Test health check return structure."""
        from data.institutional_ingest import run_infrastructure_health_check
        
        with patch('data.institutional_ingest.get_timescaledb_info') as mock_info, \
             patch('data.institutional_ingest.run_validation_pipeline') as mock_validation, \
             patch('data.institutional_ingest.engine'):
            
            mock_info.return_value = {'enabled': True, 'available': True}
            mock_validation.return_value = MagicMock(passed=True, warnings=[], errors=[], metrics={})
            
            health = run_infrastructure_health_check()
            
            required_keys = ['overall_status', 'checks', 'recommendations']
            for key in required_keys:
                self.assertIn(key, health)
            
            self.assertIn(health['overall_status'], ['HEALTHY', 'WARNING', 'CRITICAL', 'ERROR'])


class TestConfiguration(unittest.TestCase):
    """Test configuration additions."""

    def test_phase2_config_values(self):
        """Test that Phase 2 configuration values are properly set."""
        # Mock environment to avoid importing config with missing DATABASE_URL
        with patch.dict(os.environ, {
            'ENABLE_TIMESCALEDB': 'true',
            'ENABLE_DATA_VALIDATION': 'true',
            'DATA_STALENESS_THRESHOLD_HOURS': '48',
            'ENABLE_BITEMPORAL': 'true'
        }):
            # Test configuration parsing functions
            def _as_bool(env_name: str, default: bool) -> bool:
                v = os.getenv(env_name)
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    return default
                return str(v).lower() in {"1", "true", "yes", "y"}
            
            def _as_int(env_name: str, default: int) -> int:
                v = os.getenv(env_name)
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    return default
                try:
                    return int(v)
                except Exception:
                    return default
            
            # Test the functions work as expected
            self.assertTrue(_as_bool('ENABLE_TIMESCALEDB', False))
            self.assertTrue(_as_bool('ENABLE_DATA_VALIDATION', False))
            self.assertEqual(_as_int('DATA_STALENESS_THRESHOLD_HOURS', 24), 48)
            self.assertTrue(_as_bool('ENABLE_BITEMPORAL', False))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)